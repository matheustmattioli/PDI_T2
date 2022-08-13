# Código para implementação do segundo trabalho prático da disciplina
# Processamento Digital de Imagens 2022/1
# Feito pelos alunos:
#   - Lucas Machado Cid          - RA: 769841
#   - Matheus Teixeira Mattioli  - RA: 769783
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import rectangle_perimeter  # Biblioteca scikit-image


def pearsonCorrelation(img, obj):    
    # Função para realizar um template matching através da correlação de pearson.
    # Essa correlação funciona através de uma equação matemática parecida com a
    # vista em aula com o acrescimento de médias e desvios padrões.
    num_rows, num_cols = img.shape
    num_rows_obj, num_cols_obj = obj.shape   

    half_num_rows_obj = num_rows_obj//2        # O operador // retorna a parte inteira da divisão
    half_num_cols_obj = num_cols_obj//2

    # Cria imagem com zeros ao redor da borda.
    img_padded = np.pad(img, ((half_num_rows_obj,half_num_rows_obj),
                             (half_num_cols_obj,half_num_cols_obj)), 
                             mode='reflect')
    
    img_diff = np.zeros((num_rows, num_cols))
    for row in range(num_rows):
        for col in range(num_cols):
            # patch é a região de img de mesmo tamanho que obj e centrada em (row, col)
            patch = img_padded[row:row+num_rows_obj, col:col+num_cols_obj]
            # Os comandos abaixo calculam a correlação de Pearson entre cada valor
            # dos arrays 2D patch e obj.
            divisor = np.sqrt(np.sum((patch - np.mean(patch))**2) * np.sum((obj - np.mean(obj))**2))
            diff_region = ((patch - np.mean(patch)) * (obj - np.mean(obj)))/divisor
            img_diff[row, col] = np.sum(diff_region)

    # Os valores de img_diff estão - 1 e 1, com 1 representando 
    # um casamento perfeito entre o centro do obj e a posição de img.

    return img_diff


def find_max(img):
    # Encontra posição do valor maximo de img
    # Usado para encontrar o valor máximo no grid img_diff
    # que representa o melhor casamento possivel entre centro obj e posição em img
    
    ind_max = np.argmax(img)     # Retorna índice de maior valor considerando img como um array 1D
    row_max = ind_max//img.shape[1]
    col_max = ind_max - row_max*img.shape[1]
    index = (row_max, col_max)
    max_val = img[index]
    
    return max_val, index

def draw_square(img_g, center, size):
    # Função para desenhar um quadrado branco ao redor do objeto na imagem global.
    upper_left_p = (center[0] - size[0]//2, center[1] - size[1]//2)
    coords = rectangle_perimeter(upper_left_p, extent=size)

    img_box = img_g.copy()
    for i in range(len(coords[0])):
        x = min(max(0, coords[0][i]), img_g.shape[0] - 1)
        y = min(max(0, coords[1][i]), img_g.shape[1] - 1)
        img_box[x, y] = 255
    # img_box[coords] = 255
    return img_box

def rgb2gray(img):
    # Função passada pelo Professor para retirar os canais de cores de uma imagem colorida,
    # transformando em escala de cinza.
    img_gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    return img_gray

def templateMatchingCompare(matchingObjects):
    for image in matchingObjects:
        plt.figure(figsize=[15,15])
        separator = "/"
        img_g = plt.imread("images" + separator + image["name_global"])
        img_o = plt.imread("objects" + separator + image["name_object"])
        file_img_name = "." + separator + "template_matching" + separator + image["name_global"].split(".")[0].split("_global")[0] + "_" + image["name_object"].split(".")[0] + ".jpg"
        
        if(image["hasColor"]):
            img_g = rgb2gray(img_g)
            img_o = rgb2gray(img_o)

        plt.subplot(2, 2, 1)
        plt.imshow(img_g)

        plt.subplot(2, 2, 2)
        plt.imshow(img_o)
        
        plt.subplot(2, 2, 3)

        _, index = find_max(pearsonCorrelation(img_g, img_o))

        img_square = draw_square(img_g, index, img_o.shape)
        plt.figure(figsize=[10,10])
        plt.imshow(img_square, 'gray')
       
        plt.savefig(file_img_name, bbox_inches='tight')


imagesStrings = [
    {"name_global": "boys_global.jpg", "name_object": "boys.jpg", "hasColor": True},
    {"name_global": "coqueiro_global.jpg", "name_object": "coqueiro.jpg", "hasColor": True},
    {"name_global": "coqueiro_global.jpg", "name_object": "canto.jpg", "hasColor": True},
    {"name_global": "coqueiro_global.jpg", "name_object": "lua.jpg", "hasColor": True},
    {"name_global": "trashcanglobal.jpg", "name_object": "trashcan.jpg", "hasColor": True},
    {"name_global": "pessoa_global.jpg", "name_object": "pessoa.jpg", "hasColor": True},
    {"name_global": "imagem_global.tiff", "name_object": "gato.tiff", "hasColor": False},
    {"name_global": "coqueiro_baixocontrast.jpg", "name_object": "coqueiro.jpg", "hasColor": True},
    {"name_global": "coqueiro_baixocontrast.jpg", "name_object": "lua.jpg", "hasColor": True},
    {"name_global": "coqueiro_baixocontrast.jpg", "name_object": "canto.jpg", "hasColor": True},
    {"name_global": "coqueiro_escura.jpg", "name_object": "coqueiro.jpg", "hasColor": True},
    {"name_global": "coqueiro_escura.jpg", "name_object": "lua.jpg", "hasColor": True},
    {"name_global": "coqueiro_escura.jpg", "name_object": "canto.jpg", "hasColor": True},
    {"name_global": "coqueiro_qsesemcontraste.jpg", "name_object": "coqueiro.jpg", "hasColor": True},
    {"name_global": "coqueiro_qsesemcontraste.jpg", "name_object": "lua.jpg", "hasColor": True},
    {"name_global": "coqueiro_qsesemcontraste.jpg", "name_object": "canto.jpg", "hasColor": True},
]


templateMatchingCompare(imagesStrings)


