from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple, WRITES_IDX_MAP,
)

class MyCheckpointer(BaseCheckpointSaver):
    # async def aput(
    #     self,
    #     config: RunnableConfig,
    #     checkpoint: Checkpoint,
    #     metadata: CheckpointMetadata,
    #     new_versions: ChannelVersions,
    # ) -> RunnableConfig:
    #     ...

    async def aput(self, config, checkpoint, metadata, new_versions):
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        parent_id = config["configurable"].get("checkpoint_id")

        type_, blob = self.serde.dumps_typed(checkpoint)
        serialized_metadata = self.serde.dumps_typed(metadata)

        await self.db.execute(
            "INSERT INTO checkpoints (...) VALUES (...)",
            thread_id, checkpoint_ns, checkpoint_id, parent_id,
            type_, blob, *serialized_metadata,
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    # async def aput_writes(
    #     self,
    #     config: RunnableConfig,
    #     writes: Sequence[tuple[str, Any]],
    #     task_id: str,
    #     task_path: str = "",
    # ) -> None:
    #     ...

    async def aput_writes(self, config, writes, task_id, task_path=""):
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        rows = []
        for idx, (channel, value) in enumerate(writes):
            type_, blob = self.serde.dumps_typed(value)
            final_idx = WRITES_IDX_MAP.get(channel, idx)
            rows.append((thread_id, checkpoint_ns, checkpoint_id,
                         task_id, task_path, final_idx, channel, type_, blob))

        await self.db.executemany("INSERT INTO writes (...) VALUES (...)", rows)

    # async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
    #     ...

    async def aget_tuple(self, config):
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if checkpoint_id:
            row = await self.db.fetchone(
                "SELECT * FROM checkpoints "
                "WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?",
                thread_id, checkpoint_ns, checkpoint_id,
            )
        else:
            row = await self.db.fetchone(
                "SELECT * FROM checkpoints "
                "WHERE thread_id=? AND checkpoint_ns=? "
                "ORDER BY checkpoint_id DESC LIMIT 1",
                thread_id, checkpoint_ns,
            )

        if row is None:
            return None

        writes = await self.db.fetchall(
            "SELECT task_id, channel, type, value FROM writes "
            "WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=? "
            "ORDER BY task_id, idx",
            thread_id, checkpoint_ns, row["checkpoint_id"],
        )
        pending_writes = [
            (w["task_id"], w["channel"], self.serde.loads_typed((w["type"], w["value"])))
            for w in writes
        ]

        checkpoint = self.serde.loads_typed((row["type"], row["blob"]))
        metadata = self.serde.loads_typed((row["metadata_type"], row["metadata"]))

        parent_config = None
        if row["parent_checkpoint_id"]:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": row["parent_checkpoint_id"],
                }
            }

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": row["checkpoint_id"],
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    # async def alist(
    #     self,
    #     config: RunnableConfig | None,
    #     *,
    #     filter: dict[str, Any] | None = None,
    #     before: RunnableConfig | None = None,
    #     limit: int | None = None,
    # ) -> AsyncIterator[CheckpointTuple]:
    #     ...
    #     yield  # make this an async generator

    async def alist(
            self,
            config: RunnableConfig | None,
            *,
            filter: dict[str, Any] | None = None,
            before: RunnableConfig | None = None,
            limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        # 1. 解析基础分区条件
        thread_id = None
        ns = ""
        if config and "configurable" in config:
            cfg = config["configurable"]
            thread_id = cfg.get("thread_id")
            ns = cfg.get("checkpoint_ns", "")

        if not thread_id:
            return

        # 2. 拼接SQL基础语句
        sql_parts = ["SELECT * FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ?"]
        params = [thread_id, ns]

        # before 过滤：只查询早于指定checkpoint_id的历史快照
        before_cp_id = None
        if before and "configurable" in before:
            before_cp_id = before["configurable"].get("checkpoint_id")
        if before_cp_id:
            sql_parts.append("AND checkpoint_id < ?")
            params.append(before_cp_id)

        # filter 元数据过滤（metadata是序列化blob，简单场景可忽略；高级可存单独JSON字段用于索引）
        if filter:
            # 简易实现：仅演示，生产建议metadata单独存JSON字段用于索引
            pass

        # 倒序：最新快照在前
        sql_parts.append("ORDER BY checkpoint_id DESC")

        # 分页limit
        if limit is not None:
            sql_parts.append("LIMIT ?")
            params.append(limit)

        full_sql = " ".join(sql_parts)
        rows = await self.db.fetchall(full_sql, *params)

        # 逐行组装CheckpointTuple并异步yield
        for row in rows:
            # 复用aget_tuple中读取writes、反序列化逻辑
            writes_rows = await self.db.fetchall(
                "SELECT task_id, channel, type, value FROM writes "
                "WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=? "
                "ORDER BY task_id, idx",
                thread_id, ns, row["checkpoint_id"],
            )
            pending_writes = [
                (w["task_id"], w["channel"], self.serde.loads_typed((w["type"], w["value"])))
                for w in writes_rows
            ]

            checkpoint = self.serde.loads_typed((row["type"], row["blob"]))
            metadata = self.serde.loads_typed((row["metadata_type"], row["metadata"]))

            parent_config = None
            if row["parent_checkpoint_id"]:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": ns,
                        "checkpoint_id": row["parent_checkpoint_id"],
                    }
                }

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": ns,
                        "checkpoint_id": row["checkpoint_id"],
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=pending_writes,
            )

    # async def adelete_thread(self, thread_id: str) -> None:
    #     ...
    # 也可以软删除is_deleted字段
    async def adelete_thread(self, thread_id: str) -> None:
        # 先删附表writes（外键依赖，无外键可交换顺序）
        await self.db.execute(
            "DELETE FROM writes WHERE thread_id = ?",
            thread_id
        )
        # 再删主快照表
        await self.db.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?",
            thread_id
        )