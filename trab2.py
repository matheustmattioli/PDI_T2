import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import rectangle_perimeter  # Biblioteca scikit-image


def quadratic_difference(img, obj):
    num_rows, num_cols = img.shape
    num_rows_obj, num_cols_obj = obj.shape   

    half_num_rows_obj = num_rows_obj//2        # O operador // retorna a parte inteira da divisão
    half_num_cols_obj = num_cols_obj//2

    # Cria imagem com zeros ao redor da borda. Note que ao invés de adicionarmos 0, seria mais 
    # preciso calcularmos a diferença quadrática somente entre pixels contidos na imagem.
    img_padded = np.pad(img, ((half_num_rows_obj,half_num_rows_obj),
                             (half_num_cols_obj,half_num_cols_obj)), 
                             mode='reflect')
    
    img_diff = np.zeros((num_rows, num_cols))
    for row in range(num_rows):
        for col in range(num_cols):
            # patch é a região de img de mesmo tamanho que obj e centrada em (row, col)
            patch = img_padded[row:row+num_rows_obj, col:col+num_cols_obj]
            # Utilizando numpy, o comando abaixo calcula a diferença entre cada valor
            # dos arrays 2D patch e obj
            divisor = np.sqrt(np.sum((patch - np.mean(patch))**2) * np.sum((obj - np.mean(obj))**2))
            diff_region = ((patch - np.mean(patch)) * (obj - np.mean(obj)))/divisor
            img_diff[row, col] = np.sum(diff_region)

    return img_diff


def find_max(img):
    '''Encontra posição do valor maximo de img'''
    
    ind_max = np.argmax(img)     # Retorna índice de maior valor considerando img como um array 1D
    row_max = ind_max//img.shape[1]
    col_max = ind_max - row_max*img.shape[1]
    index = (row_max, col_max)
    max_val = img[index]
    
    return max_val, index

def draw_square(img_g, center, size):
    '''Desenha um quadrado em uma cópia do array img_g. center indica o centro do quadrado
       e size o tamanho.'''
    
    half_num_rows_obj = size[0]//2
    half_num_cols_obj = size[1]//2
    
    upper_left_p = (center[0] - size[0]//2, center[1] - size[1]//2)
    coords = rectangle_perimeter(upper_left_p, extent=size)

    img_box = img_g.copy()
    img_box[coords] = 255
    
    return img_box

# img = np.array([[5, 2, 0, 1, 4],
#                 [2, 4, 1, 3, 2],
#                 [3, 2, 3, 2, 4],
#                 [0, 2, 3, 4, 5],
#                 [2, 1, 0, 2, 3]])

# img_obj = np.array([[2, 3, 2],
#                     [2, 3, 4],
#                     [1, 0, 2]])

# img_diff = quadratic_difference(img, img_obj)
# print(img_diff)

img_g = plt.imread('imagem_global.tiff')
img_g = img_g.astype(float)
img_o = plt.imread('gato.tiff')
img_o = img_o.astype(float)


img_diff = quadratic_difference(img_g, img_o)

smallest_val, index = find_max(img_diff)

img_square = draw_square(img_g, index, img_o.shape)
plt.figure(figsize=[10,10])
plt.imshow(img_square, 'gray')
# plt.imshow(img_diff, 'gray')
plt.show()

