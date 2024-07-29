def fun1(image):
    row = len(image)
    col = len(image[0])

    result = [0] * (row * col)
    for r in range(row):
        for c in range(col):
            result[r*col + c] = image[r][c]
    return result

def fun2(image):
    row = len(image)
    col = len(image[0])

    result = [0] * (row*col)
    for i in range(row*col):
        r = i // row
        c = i % row
        result[i] = image[r][c]
    return result

def fun3(image):
    result = []
    for row in image:
        result.append(row)
    return result

def fun4(image):
    result = []
    for row in image:
        for pixel in row:
            result.append(pixel)
    return result

def fun5(image):
    row = len(image)
    col = len(image[0])

    result = []
    for c in range(col):
        for r in range(row):
            pixel = image[r][c]
            result.append(pixel)
    return result

image = [[1, 2, 3], [4, 5]]
print(fun5(image))


