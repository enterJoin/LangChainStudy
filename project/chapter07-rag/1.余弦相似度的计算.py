# 余弦相似度计算公式：a,b的余弦相似度为 ab的点积 / (a的模长+b的模长)
# ab的点积：a[1]*b[1] + a[2]*b[2] + a[...]*b[...]
# a的模长：开根号(a[1]*a[1] * a[2]*a[2])
import numpy

def dot(vector1, vector2):
    if len(vector1) != len(vector2):
        return None
    res = 0
    for i in range(len(vector1)):
        res += vector1[i] * vector2[i]

    return res

def norm(vector):
    res = 0
    for data in vector:
        res += data * data

    return numpy.sqrt(res)

def cos_similar(vector1, vector2):
    dot_value = dot(vector1, vector2)
    return dot_value / (norm(vector1) * norm(vector2))

if __name__ == '__main__':
    vector_a = [0.5, 0.5]
    vector_b = [0.8, 0.3]
    vector_c = [0.7, 0.7]

    print(cos_similar(vector_a, vector_b))
    print(cos_similar(vector_a, vector_c))
    print(cos_similar(vector_b, vector_c))