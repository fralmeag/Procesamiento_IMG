'''     
        Desarrollado por:
        Phd. Jimmy Alexander Cortes Osorio 
        Msc. Francisco Alejandro Medina 
        Ing. Juliana Gómez Osorio 
        Procesamiento de Imágenes
        Universidad Tecnológica de Pereira 
        Python 3.9 
        2022
'''         
import numpy as np
import matplotlib.pyplot as plt
import math
#Comando para generar la documentación HTML: pydoc -w LibreriaProImg
#-------------------------------------------------------------------------------
def imread(imagen):
    """ 
        Lee una imagen.
        Uso: Lee una imagen desde un archivo específico dado.
            La sintaxis para su uso es:
            imread('Imagen')
        Parámetros:
            Imagen--es el nombre del archivo a leer. 
            El uso de comillas simples (') delimita el nombre del archivo. En general, se pueden leer imágenes en
            cualquier formato, sea jpg, png, xpm, ppm.
        Ejemplo:
            1.Para leer una imagen se usa la siguiente instrucción, imread. Se lee una imagen
                llamada “cartagena.jpg”:
                A = plt.imread('cartagena.jpg')
    """
    matriz = plt.imread(imagen)
    return matriz
#-----------------------------------------------------------------------------------------------
def imshow(RGB):
    """ 
        Muestra una imagen RGB en python.
        Uso: Muestra la imagen en una figura gráfica en python, donde I está en escala de grises, 
            RGB (color verdadero) o binarizada. Para las imágenes binarias, imshow(I) muestra pixeles 
            con valor 0 (cero) como negro y 1 (uno) como blanco.
        La sintaxis para su uso es:
            imshow(RGB)
        Parámetros:
            RGB -- imagen RGB a mostrar contenida en una variable A cualquiera.
        Ejemplo:
            1. Para la visualización de una imagen, se lee la imagen, se almacena en una
                variable “A” y se usa la instrucción imshow. Se muestra una imagen
                llamada “cartagena.jpg”:
                A = imread('cartagena.jpg')
                imshow(A)
    """
    l =RGB.ndim
    if l == 3:
        plt.imshow(RGB)
        plt.show()
    else:
        plt.imshow(RGB, cmap='gray')
        plt.show()
    plt.axis('off')
#---------------------------------------------------------------------------------------------------
def rgb2gray(RGB):
    """ 
        Convierte imagen RGB en escala de grises.
        Uso: rgb2gray convierte una imagen de color RGB a imagen en escala de grises.
        La sintaxis para su uso es:
            rgb2gray(I)
            Donde I es la imagen RGB a convertir en escala de grises.
        Parámetros:
            RGB -- imagen a color
        Ejemplo:
            1. Se lee una imagen y se almacena en la variable A.
                A = imread('herramientas.jpg')
            2. Se cambia la imagen a escala de grises y se almacena en la variable "gris".
                gris = rgb2gray(A)
            3. Se muestra la imagen en escala de grises.
                imshow(gris)
    """

    r= RGB[:,:,0]
    g= RGB[:,:,1]
    b= RGB[:,:,2]
    gray = (0.299*r)+(0.587*g)+(0.114*b)
    return gray
#---------------------------------------------------------------------------------------------------
def imcrop(I,S):
    """ 
        Recorta una imagen entre 2 puntos específicos.
        Uso: my_imcrop recorta una imagen RGB o binarizada entre unas coordenadas específicas.
        La sintaxis para su uso es:
            imcrop(I, S)
        Parámetros:
            I -- imagen RGB o binarizada a recortar
            S -- coordenadas iniciales y finales de x, y. formato [x, y, l, h]
        Ejemplo:
            1. Se lee una imagen llamada "cartagena.jpg" de dimensiones 302x538 y se
                almacena en una variable "A".
                A = imread('cartagena.jpg')
            2. Se usa la instrucción my_imcrop para recortar la imagen "cartagena.jpg",
                almacenada anteriormente en la variable “A” con dimensiones 200x250 y se
                almacena en una variable "B".
                B = imcrop(A, [1,1,199,249])
            3. Se usa la instrucción imshow, para visualizar la imagen recortada.
                imshow(B)
    """
    x=S[0]
    y=S[1]
    w=S[2]
    h=S[3]
    c=I[x:x+w, y:y+h, :]
    return c
#---------------------------------------------------------------------------------------------
def imresize(I, s):
    """ 
        Cambia el tamaño.
        Uso:Cambia el tamaño o realiza el escalamiento de una imagen.
            La sintaxis para su uso es:
            imresize(I, s)
        Parámetros:
            I-- es la imagen para cambiar o hacer escalamiento.
            s-- es la escala a la cual se quiere ajustar.
            Cuando la entrada “m” es un vector de 2 posiciones, mls y mcols es el valor
            del número de las y columnas respectivamente del tamaño deseado.
        Ejemplo:
            1.Se lee una imagen llamada “cartagena.jpg” y se almacena en una
                variable “A”:
                A = imread('cartagena.jpg')
            2.Se usa la instrucción rgb2gray para cambiar la imagen
                “cartagena.jpg” a escala de grises y se almacena en una variable “B”:
                B = rgb2gray(A)
            3.Se usa la instrucción imresize para reducir la imagen a un 50 % de su
                escala normal y se almacena en una variable “C”.
                C = imresize(B,0.5)
            4.Se usa la instrucción imshow, para visualizar la imagen reducida:
                imshow(C)
    """
    I = I.astype(int)
    [r, c] = I.shape
    l = I.ndim
    data2 = type(s).__name__
    if((data2 != 'float' and data2 != 'list') or len(s) > 2):
        print("Error imresize: La segunda entrada debe ser un vector o un escalar")
    else:
        sx = s[0]
        if len(s) == 1:
            sy = sx
        else:
            sy = s[1]
            sx = sx/r
            sy = sy/c
        rp = int(r*sx)
        cp = int(c*sy)
        sm = [[sx, 0],[0,sy]]
        It = np.linalg.inv(sm).astype(int)
        In = np.zeros((round(rp),round(cp)), dtype=int)
        for k in range(1, l):
            for i in range(1, rp):
                for j in range(1, cp):
                    XP = It * [i,j]
                    xp = XP[0,0]
                    if xp == 0:
                        xp = 1
                    yp = round(XP[1,1])
                    if(yp == 0):
                        yp = 1
                    In[i,j] = I[xp,yp]
        return In
#------------------------------------------------------------------------------------------
def imtranslate(I,T,varargin):
    """ 
        Realiza la traslación de una imagen.
        Uso: imtranslate realiza la traslación de una imagen según las coordenadas entregadas. 
        La sintaxis para su uso es:
            imtranslate(I, t, varagin)
        Parámetros:
            I -- imagen que se va a trasladar
            t -- valor entero de pixeles a trasladar, sintaxis [x, y] 
            varagin -- tipo de traslación, valores full y same
            El parámetro varagin tiene las siguientes entradas tipo string:
            'full': La imagen se traslada y genera un zero-pad para no recortar la imagen.
            'same': La imagen de salida tiene el mismo tamaño de la imagen de entrada (defecto).
        Ejemplo:
            1. Se lee una imagen llamada "cartagena.jpg" y se almacena en una variable "A".
                A = imread('cartagena.jpg')
            2. Se traslada la imagen simétricamente, es decir 100 píxeles en las filas y
                columnas y se usa la instrucción imshow, para visualizar la imagen
                trasladada.
                C = imtranslate(A, [100],'full')
                imshow(C)
    """  
    [n,m,l]= I.shape
    salida='same'
    op_salidas=['same','full']
    if varargin != '':
        if varargin not in op_salidas:
            print ("Error imtranslate: Opcion invalida, las posibles son: full y same")
        if varargin == 'full':
            salida = 'full'
    ty=T[0]
    if len(T) == 1:
        tx = ty 
    else:
        tx=T[1] 
    In=np.zeros([n+tx,m+ty, l], dtype=int)
    In[tx:, ty:, :] = I
    if salida == 'same':
        In = imcrop(In, [1,1,n-1,m-1])
    else:
        In = In
    return In
#----------------------------------------------------------------------------------------------
def imhist(RGB):
    """ 
        Histograma de una imagen.
        Uso: Calcula el histograma de una imagen.
            La sintaxis para su uso es:
            imhist(RGB)
        Parámetros: 
            RGB-- es la imagen.
        Ejemplo:
            1.Se lee una imagen llamada “cartagena.jpg” y se almacena en una variable
                “A”:
                A = imread('cartagena.jpg')
            2.Se extrae la capa roja de la imagen “A”.
                capa_R = A (:, :, 1)
            3.Se usa la instrucción my_imhist para visualizar la matriz de la capa de la
                siguiente manera:
                print(imhist(capa_R))
    """
    hist, bins = np.histogram(RGB, np.arange(0,257), density=True)
    return hist
#-----------------------------------------------------------------------------------------------------------------
def imhist2(RGB):
    """ 
        Matriz de una imagen.
        Uso: Calcula la matiz del histograma de una imagen.
        La sintaxis para su uso es:
            imhist2(RGB)
        Parámetros:
            RGB-- es la imagen a color y se extraerá solo la capa R con el siguiente ejemplo.
        Ejemplo:
            1.Se lee una imagen llamada “cartagena.jpg” y se almacena en una variable
                “A”:
                A = imread('cartagena.jpg')
            2.Se extrae la capa roja de la imagen “A”.
                capa_R = A (:, :, 1)
            3.Se usa la instrucción my_imhist para visualizar la matriz de la capa de la
                siguiente manera:
                print(my_imhist(capa_R))
    """
    h, bins = np.histogram(RGB, bins=255)
    plt.title('Histograma')
    plt.axis([0, 255, 0, 0.3*max(h)])
    plt.grid(True)
    plt.bar(bins[0:-1], h, width=2)
    plt.plot(bins[0:-1], h, color = 'blue', )
    plt.show()
    plt.clf()
    return h
#--------------------------------------------------------------------------------------
def mean2(I):
    """
        Calcula promedio de una matriz de una imagen.
        Uso:Calcula el promedio o media de los elementos de una matriz de una
            imagen 2D y hace referencia al brillo de la misma.
        La sintaxis para su uso es:
            mean2(I, m)
        Parámetros:
            I--es una matriz de tipo “double” o de cualquier clase de enteros.
        Ejemplo:
            1.Se lee una imagen llamada “manzana.jpg” y se guarda en una variable “A”:
                A = imread('manzana.jpg')
            2.Se usa la instrucción rgb2gray para convertir la imagen “manzana.jpg” a
                escala de grises y se almacena en una variable “B”:
                B = rgb2gray(A)
            3.Se usa la instrucción mean2 para calcular el brillo de “B” y se almacena en una
                variable “C”.
                C = mean2(B)
    """
    promedio = 0
    l = I.ndim
    if l == 3:
        print("Error mean2: La imagen debe estar en escala de grises")
    else:
        promedio = np.mean(I[:])
    return promedio
#-----------------------------------------------------------------------------
def std2(I):
    """
        Calcula la desviación estándar de una matriz de una imagen.
        Uso: Calcula la desviación estándar de los elementos de una matriz de una imagen estándar 2D 
            y hace referencia al contraste de la misma.
        La sintaxis para su uso es:
            std2(I)
        Parámetros:
            I-- es la matriz numérica.
        Ejemplo:
            1.Se crea una matriz numérica:
                A = [4, 7 , 8 ; 7, 9, 0; 5, 6, 7]
            2.Se calcula la desviación estándar de la matriz.
                Des = std2(A)
    """
    sd = np.std(I, ddof=1)
    return sd
#------------------------------------------------------------------------------------
def imabsdiff(a,b):
    """ 
        Diferencia absoluta.
        Uso: Calcula la diferencia absoluta de dos imágenes de mismo tamaño.
        La sintaxis para su uso es:
            imabsdiff(a,b)
        Parámetros:
            a y b --son reales, con el mismo tamaño y clase.
            Resta cada elemento en la matriz “a” con el correspondiente elemento de la matriz b.
        Ejemplo:
            1.Se lee una imagen llamada “fondo.jpg” y se almacena en una variable “A”:
                A = imread('fondo.jpg')
            2.Se lee una imagen llamada “pera.jpg” y se almacena en una variable “B”:
                B = imread('pera.jpg')
            3.Para ajustar la diferencia absoluta de dos imágenes se requiere un alfa y se
                multiplica “A” y “B” por el 50 %:
                alpha = 0.5
                Ap = (alpha*A)
                Bp = (1−alpha)*B
            4.Se calcula la diferencia absoluta de las dos imágenes y se almacena el resultado
                en una variable “dif”
                dif = imabsdiff(Ap,Bp)
            5.Se muestra la diferencia absoluta de las dos imágenes con la instrucción
                imshow.
                imshow(dif)
    """
    [r1,c1,l1]= a.shape
    [r2,c2,l2]= b.shape 
    data1 = type(a).__name__
    data2 = type(b).__name__
    if (r2==1 and c2==1 and l2==1): 
        r2=r1
        c2=c1
        l2=l1
        data2=data1
    if (r1 != r2 or c1 != c2 or l1 != l2 or data1 != data2):
        print ("Error imabsdiff: Las imágenes de entrada deben tener el mismo tamaño y formato")
    a= a.astype(float)
    b= b.astype(float)
    s= abs(a-b)  
    return s
#-------------------------------------------------------------------------------------------------
def imadd(a,b):
    """ 
        Suma dos imágenes.
        Uso:La suma de dos imágenes o una imagen por una constante del mismo tamaño.
        La sintaxis para su uso es:
            imadd(a, b)
        Parámetros:
            a y b-- son imágenes.
            Suma cada elemento en la matriz “a” con el correspondiente elemento de la matriz
            “b”. “a” y “b” son reales, con el mismo tamaño y clase, o “b” es un escalar de tipo
            “double”.
            La matriz de salida “s”, tiene el mismo tamaño y clase si “a” y “b” son iguales.
        Ejemplo:
            1.Se lee una imagen llamada “pera.jpg” y se almacena en una variable “A” y se lee
                una imagen llamada “fondo.jpg” y se almacena en una variable “B”; y ambas
                variables se dividen por 255 para que la matriz quede en un rango entre 0 y 1:
                A = imread('pera.jpg')/255
                B = imread('fondo.jpg')/255
            2.Para el ajuste de la suma de dos imágenes se requiere un alfa:
                alpha = 0.5
                C = alpha*A
                D= (1−alpha)*B
            3.Se suman ambas imágenes y se almacena el resultado en una variable “imadd”
                imadd = imadd(C, D)
            4.Se visualiza la suma de las dos imágenes con la instrucción imshow.
                imshow(imadd)
    """
    [r1,c1,l1]= a.shape
    [r2,c2,l2]= b.shape
    data1 = type(a).__name__
    data2 = type(b).__name__
    if data2 == 'int' or data2 == 'float':
        r2 = 1
        c2 = 1
        l2 = 1
    else:
        [r2,c2,l2] = b.shape

    if (r2==1 and c2==1 and l2==1):
        r2=r1
        c2=c1
        l2=l1
        data2=data1    
    if (r1 != r2 or c1 != c2 or l1 != l2 or data1 != data2):
        print ("Error imadd: Las imágenes de entrada deben tener el mismo tamaño y formato")
    s=a+b  
    return s
#----------------------------------------------------------------------------------------------
def imdivide(a,b):
    """
        División de dos imágenes.
        Uso: Divide dos imágenes, o una imagen por una constante.
        La sintaxis para su uso es:
            imdivide(a,b)
        Parámetros:
            a y b-- son reales, con el mismo tamaño y clase, o “b” es un escalar de tipo
            “double”.
            Divide cada elemento en la matriz “a” con el correspondiente elemento de la matriz
            “b”. 
            La matriz de salida “s”, tiene el mismo tamaño y clase si “a” y “b” son iguales.
        Ejemplo:
            1.Se lee una imagen llamada “pera.jpg” y se almacena en una variable “A” y se lee
                una imagen llamada “fondo.jpg” y se almacena en una variable “B”; y la variable “A”
                se divide por 255 para que la matriz quede en un rango entre 0 y 1:
                A = imread('pera.jpg')/255
                B = imread('fondo.jpg')
            2.Para ajustar la división de dos imágenes se requiere un alfa:
                alfa=10
                C=alfa*A
                D=(1-alfa)*B
            3.Se usa la instrucción imdivide para dividir las imágenes “pera.jpg” y un escalar, y
                se almacena el resultado en una variable “imdiv”
                imdiv = imdivide(A,2)
            4. Se visualiza la división de las dos imágenes con la instrucción imshow.
                imshow(imdiv)
    """
    [r1,c1,l1]= a.shape
    data1 = type(a).__name__
    data2 = type(b).__name__
    if data2 == 'int' or data2 == 'float':
        r2 = 1
        c2 = 1
        l2 = 1
    else:
        [r2,c2,l2] = b.shape
    if (r2==1 and  c2==1 and  l2==1):
      r2= r1
      c2= c1
      l2= l1
      data2=data1
    if (r1 != r2 or c1 != c2 or l1 != l2 or data1 != data2):
       print ("Error imdivide: Las imágenes de entrada deben tener el mismo tamaño y formato")
    s = a/b
    return s
#-------------------------------------------------------------------------------------------
def immultiply(a,b):
    """
        Multiplicacion de dos imágenes.
        Uso: Multiplica dos imágenes o una imagen por una constante.
            La sintaxis para su uso es:
            immultiply(a,b)
        Parámetros:
            a y b-- son reales, con el mismo tamaño y clase, o “b” es un escalar de
            tipo “double”.
            Multiplica cada elemento en la matriz “a” con el correspondiente elemento de la
            matriz b. 
            La matriz de salida “s”, tiene el mismo tamaño y clase si “a” y “b” son iguales.
        Ejemplo:
            1.Se lee una imagen llamada “pera.jpg” y se almacena en una variable “A”, y se lee
                una imagen llamada “fondo.jpg” y se almacena en una variable “B”; y ambas
                variables se dividen por 255 para que la matriz quede en un rango entre 0 y 1”:
                A = imread('pera.jpg')/255
                B = imread('fondo.jpg')/255
            2.Se visualiza las dos imágenes A y B con la instrucción imshow.
                imshow(A)
                imshow(B)
            3.Se usa la instrucción immultiply para multiplicar las imágenes “pera.jpg” y
                “fondo.jpg” y se almacena el resultado en una variable “C”.
                C = immultiply(A,5)
            4.Se visualiza la multiplicación de las dos imágenes con la instrucción imshow.
                imshow(C)
    """
    [r1,c1,l1]= a.shape
    data1 = type(a).__name__
    data2 = type(b).__name__
    if data2 == 'int' or data2 == 'float':
        r2 = 1
        c2 = 1
        l2 = 1
    else:
        [r2,c2,l2] = b.shape
        b= b.astype(float)
    if (r2==1 and c2==1 and  l2==1):
      r2= r1
      c2= c1
      l2= l1
      data2=data1
    if (r1 != r2 or c1 != c2 or l1 != l2 or  data1 != data2):
        print ('Error immultiply: Las imagenes de entrada deben tener el mismo tamaño y formato')
    a= a.astype(float)
    s = np.multiply(a,b) 
    return s
#------------------------------------------------------------------------------
def imsubtract(a,b):
    """ Resta dos imágenes.
        Uso: Resta dos imágenes o una constante de un imagen.
        La sintaxis para su uso es:
            imsubtract(a,b)
        Parámetros:
            a y b-- son imágenes, ambas imágenes deben tener el mismo tamaño y el mismo tipo de dato. 
            Cuando la entrada “b” es un valor numérico “S”, se realiza la
            resta entre la imagen “a” y el valor numérico.
        Ejemplo:
        1.Se lee una imagen llamada “fondo.jpg” y se almacena en una variable “A” y se
            lee una imagen llamada “pera.jpg”, se almacena en una variable “B”; y ambas
            variables se dividen por 255 para que la matriz quede en un rango entre “0 y 1”:
            A = imread('fondo.jpg')/255
            B = imread('pera.jpg')/255
        2.Se usa la instrucción imsubtract para restar las imágenes “fondo.jpg” y “pera.jpg”
            y se almacena el resultado en una variable “C”
            C = imsubtract(A,B)
        3.Se visualiza la resta de las dos imágenes con la instrucción imshow.
            imshow(C)
    """

    [r1,c1,l1]= a.shape
    data1 = type(a).__name__
    data2 = type(b).__name__
    if data2 == 'int' or data2 == 'float':
        r2 = 1
        c2 = 1
        l2 = 1
    else:
        [r2,c2,l2] = b.shape

    if (r2==1 and c2==1 and  l2==1): #constante
      r2= r1
      c2= c1
      l2= l1
      data2=data1
    if (r1 != r2 or c1 != c2 or l1 != l2 or  data1 != data2):
        print ('Error imsubtract: Las imagenes de entrada deben tener el mismo tamaño y formato')
    s = np.subtract(a, b)
    return s
#----------------------------------------------------------------------------------------
def imadjust(r,E,S,n):
    """
        Ajusta valores de una imagen.
        Uso: Ajusta los valores de intensidad de una imagen.
            La sintaxis para su uso es:
            imadjust(r, E, S, n)
        Parámetros:
            s--es la imagen que se ajustará.
            Ex-- Ajusta los valores de entrada del histograma de la imagen r y es un vector
            de dos valores numéricos entre 0 y 1 [Emin, Emax].
            Sx-- Ajusta los valores de salida del histograma de la imagen r y es un vector
            de dos valores numéricos entre 0 y 1 [Smin, Smax].
            n--Ajusta los valores de la curva del histograma de la imagen “r” y es un valor
            numérico mayor que cero y menor que 10. Se conoce como Gamma.
        Ejemplo:
            1.Se lee una imagen llamada “cartagena.jpg” y se almacena en una variable “A”.
                A = imread('cartagena.jpg')
            2.Se convierte la imagen a escala de grises y se almacena en la variable “gris”.
                gris = rgb2gray(A)
            3.Se ajusta la imagen en escala de grises y el resultado se almacena en una
                variable “S”
                S = imadjust(gris,[0.2,0.8],[0,1],1)
            4.Se imprime el resultado con la instrucción imshow:
                imshow(S)
    """

    Em= E[0]*255
    EM= E[1]*255
    Sm= S[0]*255
    SM= S[1]*255
    S= (((SM-Sm)/pow(EM-Em,n))*(pow(abs(r.astype(float)-Em),n))) + Sm
    return S
#------------------------------------------------------------------------------------------
def stretchlim(B):
    """
        Calcula limites de un histograma de una imagen.
        Uso:Calcula los límites superior e inferior del histograma de una imagen.
        La sintaxis para su uso es:
            stretchlim(B)
        Parámetros:
            B--es la imagen en escala de grises.
        Ejemplo:
            1.Se lee una imagen llamada 'cartagena.jpg' y se almacena en la variable “A”:
                A = imread('cartagena.jpg')
            2.Se convierte la imagen a escala de grises y se almacena en la variable “gris”.
                gris = rgb2gray(A)
            3.Se hace uso de los valores de ajuste de la imagen en escala de grises “Emin”
                y “Emax” para usar en la función my_imadjust y se almacena el resultado en
                la variable “S”
                S = imadjust(gris, stretchlim(gris), [0,1],1)
    """
    h = imhist(B)
    hp = h/np.sum(h)
    hac = np.zeros((1,256))
    Em = 0
    for i in range(0, 255):
        hac[0][i]=np.sum(hp[0:i])      
        if hac[0][i] <= 0.01:
            Em=i
        if hac[0][i] <= 0.99:
            EM=i
    Em = Em/255
    EM = EM/255
    E=[Em, EM]
    return E
#------------------------------------------------------------------------
def imnoise(I,tipo,porcentaje):
    """
        Ruido a una imagen.
        Uso: Añade un tipo de ruido a una imagen.
        La sintaxis para su uso es:
            imnoise(I,tipo,porcentaje)
        Parámetros:
            I--es una imagen, “tipo” es el tipo de ruido, “porcentaje” es el parámetro del ruido.
            Uso de la variable “porcentaje”:
            "salt & pepper":“porcentaje” es la densidad del ruido, es el parámetro
            numérico entre 0 y 1.
            "gaussian": “porcentaje” es la varianza del ruido, es un parámetro numérico
            entre 0 y 1.
            “speckle”: “porcentaje” es la varianza del ruido,añade ruido multiplicativo
            utilizando la ecuación, donde se distribuye uniformemente el ruido con la
            media 0 y 0,05.
        Ejemplo:
            1.Se lee una imagen llamada “cartagena.jpg” y se almacena en una variable “A”:
                A = imread('cartagena.jpg')
            2.Se convierte la imagen a escala de grises y se almacena en la variable gris:
                gris = rgb2gray(A)
            3.Se usa la instrucción my_imnoise para agregar ruido de tipo 'salt & pepper' y se
                almacena en una variable “C”.
                C = imnoise(gris, 'salt & pepper', 0.1)
            4.Se usa la instrucción my_imnoise para agregar ruido de tipo 'gaussian' y se
                almacena en una variable “D”.
                D = imnoise(gris, 'gaussian' ,0.09)
            5.Se usa la instrucción my_imnoise para agregar ruido de tipo 'speckle' y se almacena
                en una variable “D”.
                E = imnoise(gris, 'speckle' ,0.05)
            6. Se usa la instrucción imshow, para visualizar las tres imágenes con ruido:
                imshow(C)
                imshow(D)
                imshow(E)
    """
    In = I
    [f, c ]= I.shape
    tipologia = tipo.lower()
    if tipologia == 'salt&pepper':
        puntos = f*c*porcentaje
        i = 1
        for i in range(int(puntos)):
            x = np.random.randint(1,f)
            y = np.random.randint(1,c)
            In[x,y] = 255*np.random.rand(1)
            In = In.astype(int)
    if tipologia == 'gaussian':
        sigma = porcentaje
        ruido = np.random.randn(f,c)*sigma*128
        In =I+ruido
    if tipologia == 'speckle':
        sigma = porcentaje
        ruido = np.random.random((f,c))*sigma*64
        In = I*ruido+I
    return In
#------------------------------------------------------------------------------------------
def medfilt2(Gris,W):
    """
        Filtrado promedio de una imagen.
        Uso: Realiza el filtrado promedio a una imagen en 2D.
        La sintaxis para su uso es:
            medfilt2(Gris,w)
        Parámetros:
            Gris-- es la imagen en escala de grises.
            w-- es un vector de dos valores numéricos que define el tamaño de la ventana, es [3,3].
        Ejemplo:
            1.Se lee una imagen llamada cartagena.jpg y se almacena en una variable I:
                I = imread('cartagena.jpg')
            2.Se conviere la imagen a escala de grises y se almacena en una variable “B”:
                B = rgb2gray(I)
            3.Se usa la instrucción imnoise para agregar tipo de ruido 'salt & pepper' y
                se almacena en una variable “C”.
                C = imnoise(B,'salt & pepper',0.02)
            4.Se usa la instrucción medfilt2 para filtrar el ruido de tipo 'salt & pepper' y se
                almacena en una variable “D”:
                D = medfilt2(C,[3,3])
            5.Se visualiza las dos imágenes con la instrucción imshow.
                imshow(C)
                imshow(D)
    """
    n = W[0]
    m = W[1]
    iniF = int((n+1)/2)
    iniC = int((m+1)/2)
    finF = int(iniF-1)
    finC = int(iniC-1)
    Ipad = np.pad(Gris.astype(int), [finF, finC], 'edge')
    [F, C] = Ipad.shape
    T = np.zeros((F,C),dtype=int)
    for i in range(iniF, F-finF):
        for j in range(iniC, C-finC):
            V = Ipad[i-finF:i+finF+1,j-finC:j+finC+1]
            ordenar = np.sort(V, axis=None)
            central=int((n*m+1)/2)
            T[i,j]= ordenar[central]
    T = T[iniF:F-finF,iniC:C-finC]
    return T
#--------------------------------------------------------------------------
def ordfilt2(Gris,termino,K):
    """
        Filtrado estadístico de una imagen.
        Uso: Realiza un filtrado estadístico a una imagen 2D.
        La sintaxis para su uso es:
            ordifilt2(Gris,termino,K)
        Parámetros:
            “Gris--es la imagen en escala de grises, “termino” es el elemento que se reemplaza de la 
            ventana el cual es un parámetro numérico entero, 
            K-- es una matriz “double”.
        Ejemplo:
            1.Se lee la imagen 'cartagena.jpg' y se almacena en la variable “I”.
                I = imread('cartagena.jpg')
            2.Se convierte la imagen a escala de grises y se almacena en la variable “Ic”.
                Ic = rgb2gray(I)
            3.Se usa la instrucción imnoise para agregar tipo de ruido “salt & pepper” a la imagen “Ic” 
                y se almacena en la variable “Ir”
                Ir = imnoise(Ic,'salt & pepper',0.02)
            4.Se realiza el filtrado mínimo 2D.
                Imed = ordfilt2(Ir,1,np.ones(3,3), dtype=int)))
            5.Se visualiza la imagen con filtrado estadístico con la instrucción imshow.
                imshow(Imed)
    """
    [n , m]= K.shape
    iniF = int((n+1)/2)
    iniC = int((m+1)/2)
    finF = int(iniF-1)
    finC = int(iniC-1)
    Ipad = np.pad(Gris.astype(int), (finF, finC), 'edge')
    Ipad = Ipad.astype(int)
    [F, C] = Ipad.shape
    T = np.zeros((F,C), dtype=int)
    for i in range(iniF,F-finF):
        for j in range(iniC, C-finC):
            V = (Ipad[i-finF:i+finF+1,j-finC:j+finC+1])*K
            ordenar = np.sort(V, axis=0)
            T[i, j] = ordenar[0][termino]
    T = T[iniF:F-finF,iniC:C-finC]
    return T
#-------------------------------------------------------------------------------------
def fspecial(**kwargs):
    """ 
        Máscara de correlación para filtrar una imagen.
        Uso:Crea un kernel (máscara) de correlación para filtrar una imagen 2D.
        La sintaxis para su uso es:
            fspecial(kwargs)
        Parámetros:
            kwargs--es el kernel.
            Manejo de la variable “kwargs”:
            En el filtro “average”, la máscara ser de “nxn” y ser una matriz de unos
            multiplicada por el recíproco del cuadrado de “n”.
            En el filtro “laplaciano”, “n” especifica el tipo de “laplaciano” a calcular
            (variable alpha).
            En el filtro “gausiano”, “n” es el parámetro correspondiente a la desviación
            estándar, pues este filtro estima el tamaño del kernel con este valor.
            El tipo de kernel debe especificarse, así:
            'average': Máscara para filtro por promedio (blur), tamaño es un vector [N,N],
            por defecto es [3,3].
            −−>kernel = fspecial ('average', tamaño)
            'laplacian’: Filtro “laplaciano”, pasa alto derivativo (detector de bordes y
            detalles), alpha es un valor entre 0 y 1, por defecto es 0.2.
            −−>kernel = fspecial ('laplacian', alpha)
            'gaussian': Filtro gaussiano pasa bajo, tamaño es un escalar y sigma un valor
            entre 0 y 1, por defecto tamaño es de [3,3] y sigma 0.5.
            −−>kernel = fspecial ('gaussian', tamaño, sigma)
            FILTROS PARA ALTOS DERIVATIVOS DIRECCIONALES:
            'sobel’: Filtro direccional que detecta líneas horizontales.
            −−>kernel = fspecial ('sobel')
            'prewitt': Filtro direccional
            −−>kernel = fspecial ('prewitt')
        Ejemplo:
            1.Se crea un kernel de filtro promedio en “K2” y así para cada uno de los tipo de
                kernel.
                K2 = fspecial(tipo= 'average’)
        Ejemplo:
                1.Se crea un kernel de filtro promedio en “K2” y así para cada uno de los tipo de
                kernel.
                K2 = fspecial(tipo= 'average’)
    """

    tipologia = kwargs['tipo']

    if tipologia == 'average':
        if len(kwargs) == 1:
            L = 3
        K = np.ones((L,L))/L**2
    if tipologia == 'gaussian':
        if len(kwargs) == 1:
            S = 0.5
            L = 3
        Limite = (L-1)/2
        [X, Y] = np.meshgrid(-Limite,Limite)
        Z = math.exp(-(np.dot(X,2)+np.dot(Y,2))/(2*(S**2)))
        K = Z/sum(Z)
    if tipologia == 'laplacian':
        if len(kwargs) == 1:
            L = 0.2
            alpha = L
        K=4/(alpha+1)*[[alpha/4, (1-alpha)/4,  alpha/4],
            [(1-alpha)/4, -1, (1-alpha)/4],
            [alpha/4, (1-alpha)/4, alpha/4]]
    if tipologia == 'log':
        if len(kwargs):
            S = 0.5
            L = 5
        Limite = (L-1)/2
        [X, Y] = np.meshgrid(-Limite,Limite)
        k = 1/(2*math.pi*S^2)*math.exp(-(np.dot(X,2)+np.dot(Y,2))/(2*(S**2)))
        ks = sum(k)
        Termin_z = np.dot(X,2) + np.dot(Y,2) - 2*(S**2)
        e = math.exp(-(np.dot(X,2) + np.dot(Y,2))/(2*(S**2)))
        f = 1/(2*math.pi*(S**6)*ks)
        Z = np.dot(f*(Termin_z), e)
        K = Z-sum(Z)/L^2
    if tipologia == 'prewitt':
        K = [[1, 1, 1],
             [0, 0, 0], 
            [-1, -1, -1]]
    if tipologia == 'sobel':
        K = [[1, 2, 1], 
            [0, 0, 0], 
            [-1, -2, -1]]
    return K
#----------------------------------------------------------------------------
def strel(tipo,forma):
    """ Elemento estructurante.
        Uso: Crea un elemento estructurante morfológico.
        La sintaxis para su uso es:
            strel('tipo', forma)
        Parámetros:
            tipo--es el tipo de elemento estructurante.
            forma--es el parámetro de configuración del elemento.
            Uso de la variable “forma”:
            'disk, 'lineV ' y 'lineH': El parámetro es un valor numérico que define el
            tamaño de la matriz.
            'square' y 'cross': El parámetro es un vector de 2 posiciones ([f,c]) o un valor
            numérico que define el tamaño de la matriz.
        Ejemplo:
            1.Se crea un elemento estructurante tipo 'cross' y 'lineV'.
                EE = strel('cross',5)
                EE1 = strel('lineV',5)
    """
    if tipo == 'disk':
        Y = 1
        a = np.arange(-forma, forma+1)
        b = np.arange(-forma, forma+1)
        X, Y = np.meshgrid(a, b)
        f = X**2 + Y**2 <= forma**2
        f = 1*f
    if tipo == 'square':
        f = np.ones((forma, forma), dtype=int)
    if tipo == 'lineV':
        a = np.zeros((forma, forma), dtype=int)
        [fils ,cols]= a.shape
        cont = 0
        for cols in range(1, forma) :
            for fils  in range(0, forma):
                X = round(forma/2)-1
                if cont == X:
                    a[fils][cols]=1            
                if cont != X:
                    a[fils][cols]=0
            cont = cont+1
        f=a 
    if tipo == 'lineH':
        a = np.zeros((forma, forma), dtype=int)
        [fils, cols]= a.shape 
        cont = 0
        for cols in range(1, forma):
            for fils in range(1, forma):
                X = round(forma/2)-1
                if cont == X:
                    a[cols] = 1
            cont = cont+1
        f=a
    if tipo == 'cross':
        a = np.zeros((forma, forma), dtype=int)
        [fils, cols] = a.shape
        cont=0
        for fils in range(1,forma):
            for cols in range(1, forma):
                X = round(forma/2)-1
                if cont == X:
                    a[fils]=1
            cont=cont+1
        [fils ,cols]= a.shape
        cont=0
        for cols in range(1, forma):
            for fils in range(0, forma):
                X = round(forma/2)-1
                if cont == X:
                    a[fils][cols]=1
            cont=cont+1
        f=a
    return f
#------------------------------------------------------------------------------------
def graythresh(RGBgray):
    """
        Umbral de una imagen.
        Uso: Calcula el umbral de una imagen utilizando el método de Otsu.
        La sintaxis para su uso es:
            graythresh(I)
        Parámetros:
            T--es un valor de intensidad normalizado que se encuentra en el intervalo de [0, 1].
            Calcula el umbral global “T” que puede ser usado para convertir una imagen de
            intensidad “I” a una imagen binaria con im2bw.
            La función my_graythresh utiliza el método de Otsu, que elige el umbral para
            minimizar la varianza intraclase de los pixeles en blanco y negro.
        Ejemplo:
            1.Se lee una imagen llamada “manzana.jpg” y se almacena en una variable “A”:
                A = imread('manzana.jpg')
            2.Se guarda la componente azul de la imagen leída y se almacena en una variable
                “B”.
                B = A(:,:,2)
            3.Se usa la instrucción graythresh para calcular el umbral y se almacena en
                una variable “T”.
                T = graythresh(B)
    """
    l = RGBgray.ndim
    [filas, cols] = RGBgray.shape
    if l==3:
        print('La imagen debe estar en escala de grises')
    else:
        h = imhist(RGBgray)
        lh = len(h)                                   
        tam = filas*cols                                                
        arr_temp = np.arange(256)
        maxV= 0
        for T in range(1,lh):                      
            Wb = np.sum(h[1:T])/tam
            Ub = np.sum(arr_temp[1:T]*h[1:T]) / np.sum(h[1:T])
            Wf = 1-Wb
            Uf = np.sum(arr_temp[T:lh]*h[T:lh]) / np.sum(h[T:lh])
            BVC = Wb * Wf * (Ub - Uf) ** 2
            if BVC >= maxV:
                maxV = BVC
                umbral= (T-1)/255
        return umbral
#-------------------------------------------------------------------------------------
def im2bw(b,T):
    """
        Conversión de una imagen a una imagen binaria.
        Uso: Convierte una imagen a una imagen binaria, basado en el umbral “T”.
        La sintaxis para su uso es:
            im2bw(b,T)
        Parámetros:
            b- es la imagen a binarizar.
            T--es el umbral.
            Produce una imagen binarizada “B” a partir de una imagen “I” en escala de grises.
        Ejemplo:
            1.Se lee una imagen llamada “manzana.jpg” y se almacena en una variable “A”:
                A = imread('manzana.jpg')
            2.Se guarda la componente azul de la imagen leída y se almacena en una variable
                “B”
                B = A(:,:,3)
            3.Se usa la instrucción my_graythresh para calcular el umbral y se almacena
                en una variable “T”.
                T = graythresh(B)
            4.Se usa la instrucción im2bw para binarizar la imagen y se almacena en una
                variable “binaria”.
                binaria = im2bw(b,T)
            5.Se visualiza la imagen binarizada con la instrucción imshow:
                imshow(binaria)
    """
    bin = (b >= T*255)
    return bin
#---------------------------------------------------------------------------------
def histeq(img):
    """
        Ecualiza el histograma.
        Uso:Ecualiza el histograma para mejorar el contraste de una imagen.
        La sintaxis para su uso es:
            histeq(img)
        Parámetros:
            img--Es una imagen en escala de grises.
        Ejemplo:
            1.Se lee una imagen y se almacena en la variable A.
                A= imread('cartagena.jpg')
            2.Se convierte la imagen a escala de grises y se almacena en la variable gris.
                gris =rgb2gray(A)
            3.Se visualiza el histograma de la imagen gris.
                imhist(gris)
            4.Se realiza la ecualización de la imagen en escala de grises
                S=histeq(gris)
                imhist(S)
    """
    h= imhist(img)
    ha = np.array([sum(h[:i+1]) for i in range(len(h))])
    he = np.uint8(255 * ha)
    r,c= img.shape[:2]
    S= np.zeros_like(img)
    for i in range(0, r):
        for j in range(0, c):
            S[i, j] = he[int(img[i, j])]
    return S
#----------------------------------------------------------------------------------------
def imfilter(Gris,K):
    """
        Realiza correlación 2D.
        Uso: Realiza filtrado 2D de imágenes bidimensionales.
        La sintaxis para su uso es:
            imfilter(Gris,K)
        Parámetros:
            Gris--Es la imagen en escala de grises.
            K--Es el tipo de correlación.
            El parametro "K" tiene las siguientes entradas tipo string:
            'full': Realiza la correlación completa.
            'same': Retorna la parte central del resultado que es del mismo tamaño de I (defecto).
            'valid': Retorna una porción de la correlación computada sin zero-padd.
        Ejemplo:
            1.Se lee una imagen llamada "cartagena.jpg" y se almacena en una variable I.
                I= imread('cartagena.jpg')
            2.Se convierte a escala de grises y se almacena en una variable Ic:
                Ic= rgb2gray(I)
            3. Se añade ruido gaussiano a la imagen Ic.
                Ir= imnoise(Ic,'gaussian',0.1)
            4. Se crea un kernel de filtro promedio en K2
                K2 = fspecial(tipo = 'average')
            5. Se realiza la correlación entre la imagen Ic y el kernel K2.
                imf2 = my_imfilter(Ir, K2)
            6. Se usa la instrucción "imshow" para mostrar la imagen.
                imshow(imf2)
    """
    [n , m] = K.shape
    iniF = int((n+1)/2)
    iniC = int((m+1)/2)
    finF = int(iniF-1)
    finC = int(iniC-1)
    Ipad = np.pad(Gris.astype(int), (finF, finC), 'edge')
    Ipad = Ipad.astype(int)
    [F, C] = Ipad.shape
    T = np.zeros((F,C), dtype=int)
    for i in range(iniF, F-finF):
        for j in range(iniC, C-finC):
            V = Ipad[i-finF:i+finF+1,j-finC:j+finC+1]
            T[i,j] = np.sum(V.flatten(order = 'C')*K.flatten(order = 'C'))
    T = T[iniF:F-finF,iniC:C-finC]
    return T
#--------------------------------------------------------------------------
def mat2gray(I, **lim):
    """
        Convierte matriz.
        Uso: Convierte la matriz numérica en una imagen en escala de grises.
        La sintaxis para su uso es:
            mat2gray(I,**lim)
        Parámetros:
            I-- Es una imagen en RGB.
            lim--Valores de la matriz.
        Ejemplo:
            1.Se crea una matriz númerica y se almacena en una variable "A".
                A=[[4,7,8],[7,9,0],[5,6,7]]
            2.Se convierte en una imagen en escala de grises y se almacena en una variable "gris"
                gris= mat2gray(A)
            3. Se visualiza la imagen en escala de grises con la instruccion "imshow".
                imshow(gris)
    """
    if len(lim) == 0:
        lim = [min(I.flatten(order = 'C')), max(I.flatten(order = 'C'))]
    else:
        lim = lim['lim']
    G = ((I-lim[0])/(lim[1]-lim[0]))
    return G
#------------------------------------------------------------------------------
def imwrite(A, filename, **kwargs):
    """
        Escribe una imagen en un archivo.
        Uso: Escribe los dato de una matriz de una imagen en un archivo.
        La sintaxis para su uso es:
            imwrite(A,filename,**kwargs)
        Parámetros:
            A--Matriz de entrada donde se encuentran los datos de la imagen.
            filename--Nombre del archivo a escribir.
            **kwargs--Recibe un numero variable de argumentos.
        Ejemplo:
            1. Se genera una imagen en escala de grises,se almacena en un una variable "a" el archivo de 
                imagen y se visualiza con la instrucción "imshow". 
                a = np.uint8(255*np.random.rand(64, 64))
                ims.imshow(a)
                imwrite(a, 'imwrite.jpg', cmapa = 'gray')
            2. Se lee el archivo de imagen.
                b = imread('imwrite.jpg')
            3. Se visualiza la imagen con la instrucción "imshow".
                plt.imshow(b)
                plt.show()
    """
    if len(kwargs) > 0:
        cmapa = kwargs['cmapa']
    plt.imsave(filename, A, cmap = cmapa)
#---------------------------------------------------------------
def hough(I):
    """
        Trasnformada de Hough
        Uso: Calcula la transformada de Houhg en una imagen binaria.
        La sintaxis para su uso es:
            hough(I)
        Parámetros:
            I--Es la imagen binarizada.
            Los parámetros opcionales son:
            'Rhomap'--modica la resolución de Rho, valor puede ser de cualquier tipo númerico.
            [P,Theta,Rho]=hough(Ibin,'RhoResolution',valor);
            'Thetamap'-- modica la resolución de Theta, valor puede ser de cualquier tipo númerico.
            [P,Theta,Rho]=hough(Ibin,'RhoResolution',valor);
            'Th'--determina el rango de valores de theta en un vector de 2 posiciones [Tmin,Tmax].
            [P,Th,Rho]=hough(Ibin,'Th',[Tmin,Tmax]);
        Ejemplo:
        1. Se lee una imagen llamada "rectas.jpg" y se almacena en la variable A.
            A = imread('rectas.jpg')
        2. Se convierte la imagen en escala de grises y se almacena en una variable Igris.
            Igris = rgb2gray(A)
        3. Se binariza la imagen,se almacena en la variable Ibin y se visualiza con la instrucción
            imshow.
            Ibin = im2bw(Igris, 0.8)
            Ibin = ~Ibin*1
            imshow(Ibin)
        4. Se calcula la transformada de Hough de la imagen Ibin y se transforma para ser 
            visualizada con la instrucción imshow.
            Th, Theta, Rho = hough(Ibin)
            Hm1 = mat.mat2gray(Th)
            imshow(Hm1)
            Hm2 = ima.imadjust(Hm1, [0,0.0824], [0, 1], 1)
            imshow(Hm2)    
    """
    f,c = I.shape
    rhomax = int(np.ceil(pow(pow(f,2)+pow(c, 2), 0.5)))
    thetamap = np.arange(-90, 91).reshape(1, 181)
    rhomap = np.arange(-rhomax, rhomax+1, dtype=int).reshape(1, 667)
    Th = np.zeros((int(2*rhomax+1), 181))
    for i in range(0, f):
        for j in range(0, c):
            if I[i, j] == 1:
                for theta in range(-90, 91):
                    rho = int(np.ceil((j*np.cos(np.radians(theta))) + (i*np.sin(np.radians(theta))))) +1
                    Th[rho+rhomax, theta+90] = Th[rho+rhomax, theta+90] + 1
    return [Th, thetamap, rhomap]
#--------------------------------------------------------------
def imerode(Ibin,SE):
    """
        Erosión.
        Uso: Erosiona una imagen.
        La sintaxis para su uso es:
            imerode(Ibin,SE)
        Parámetros:
            Ibin--Es la imagen binaria o en escala de grises.
            SE--Elemento estructurante que corresponde a una matriz.
        Ejemplo:
        1. Se lee una imagen llamada herramientas.jpg y se almacena en una variable A.
            A=imread('herramientas.jpg');
        2. Se convierte la imagen a escala de grises y se almacena en una variable B.
            B=rgb2gray(A)
        3. Se usa la instrucción  graythresh para encontrar el umbral T de la imagen 
            y se almacena en una variable T.
            T = graythresh(B)
        4. Se usa la instrucción im2bw para binarizar la imagen B y se almacena en una variable C.
            C = im2bw(B,T)
        5. Se usa la instrucción strel para crear un elemento estructurante de tipo 'disk' 
            de tamaño 3x3 y se almacena en una variable SE.
            SE = strel('disk',3)
        6. Se usa la instrucción imerode para erosionar la imagen C y almacena en una variable D.
            D = imerode(C, SE)
        7. Se visualizan las imágenes binarizada y erosianada con la instrucción imshow.
            imshow(C)
            imshow(D)
    """
    Ibin = 1*Ibin
    filt = np.ones((7, 7))
    r, c = Ibin.shape
    f, co = SE.shape
    R = r+f-1
    C = c+co-1
    In = np.zeros((R, C))
    for i in range(r):
        for j in range(c):
            In[i+1, j+1] = Ibin[i, j]
    for i in range(r):
        for j in range(c):
            k = In[i:i+f,j:j+co]
            resultado = (k == filt)
            final = np.all(resultado == True)
            if final:
                Ibin[i,j] = 1
            else:
                Ibin[i,j] = 0
    return Ibin
#--------------------------------------------------------------------------------------
def imdilate(Ibin,SE):
    """
        Dilatación.
        Uso: Dilata una imagen binaria o en escala de grises.
            La sintaxis para su uso es:
            imdilate (Ibin,SE)
        Parámetros:
            Ibin--Es la imagen binaria o en escala de grises.
            SE--Elemento estructurante que corresponde a una matriz.
        Ejemplo:
            1. Se lee una imagen llamada herramientas.jpg y se almacena en una variable A.
                A=imread('herramientas.jpg');
            2. Se convierte la imagen a escala de grises y se almacena en una variable B.
                B=rgb2gray(A)
            3. Se usa la instrucción  graythresh para encontrar el umbral T de la imagen 
                y se almacena en una variable T.
                T = graythresh(B)
            4. Se usa la instrucción im2bw para binarizar la imagen B y se almacena en una variable C.
                C = im2bw(B,T)
            5. Se usa la instrucción strel para crear un elemento estructurante de tipo 'disk' 
                de tamaño 3x3 y se almacena en una variable SE.
                SE = strel('disk',3)
            6. Se usa la instrucción imdilate para dilatar la imagen C y almacena en una variable D.
                D = imdilate(C, SE)
            7. Se visualizan las imágenes dilatada con la instrucción imshow.
                imshow(D)
    """
    l = Ibin.ndim
    if l > 2:
        print("imdilate: La imagen debe ser binaria")
    d = ~imerode(~Ibin,SE)
    return d
#-----------------------------------------------------------
def imclose(Ibin,SE):
    """
        Cerrado.
        Uso: Realiza cerrado morgológico de una imagen binaria o en escala de grises.
        La sintaxis para su uso es:
            imclose(Ibin,SE)
        Parámetros:
            Ibin--Es la imagen binaria o en escala de grises.
            SE--Elemento estructurante que corresponde a una matriz.
        Ejemplo:
            1. Se lee una imagen llamada herramientas.jpg y se almacena en una variable A.
                A=imread('herramientas.jpg');
            2. Se convierte la imagen a escala de grises y se almacena en una variable B.
                B=rgb2gray(A)
            3. Se usa la instrucción  graythresh para encontrar el umbral T de la imagen 
                y se almacena en una variable T.
                T = graythresh(B)
            4. Se usa la instrucción im2bw para binarizar la imagen B y se almacena en una variable C.
                C = im2bw(B,T)
            5. Se usa la instrucción strel para crear un elemento estructurante de tipo 'disk' 
                de tamaño 3x3 y se almacena en una variable SE.
                SE = strel('disk',3)
            6. Se usa la instrucción imclose para dilatar la imagen C y almacena en una variable D.
                D = imclose(C, SE)
            7. Se visualizan la imágen cerrada con la instrucción imshow.
                imshow(D)
    """
    l = Ibin.ndim
    di = imdilate(Ibin, SE)
    di = di*-1
    cl = imerode(di, SE)
    return cl
#------------------------------------------------------------------
def fft2(I):
    x, y = I.shape
    F = np.fft.fftn(I, (x,y))
    return F
#-----------------------------------------------------------------
def fftshift(Fw):
    Fwshift = np.fft.fftshift(Fw)
    return Fwshift
#-------------------------------------------------------------------
def ifft2(I):
    x, y = I.shape
    ift = np.fft.ifft2(I, (x,y))
    return ift
#---------------------------------------------------------------------------
def bwperim(B, conn = 4):
    """
        Perímetro de una imagen.
        Uso:Encuentra los objetos del perímetro de una imagen.
        La sintaxis para su uso es:
            bwperim(B, conn = 4)
        Parámetros:
            B--es la imagen binarizada 
            conn=4--es la imagen con contorno.
        Ejemplo:
            1. Se lee una imagen llamada “herramientas.jpg”, y almacena en una variable “RGB”:
                RGB= imread(‘herramientas.jpg’)
            2. Se cambia a escala de grises y se almacena  en una variable “gris”
                gris=rgb2gray(RGB)
            3. Se binariza la imagen con la instrucción im2bw y se almacena en una variable “binaria”.
                T=graythresh(gris)
                binaria=im2bw(gris,T)
            4. Se crea un filtrado morfológico para que la imagen esté libre de píxeles que no tengan 
                vecindad  y se muestra usando la instrucción imshow.
                EE=strel(‘square’,3);
                dilatada=imdilate(~binaria, EE)
                imshow(dilatada)
            5. Se evalúa el contorno de la región en blanco.
                bwperim(dilatada,4)
    """
    f, c = B.shape
    if conn == 4:
        conn = np.array([[0,1,0], [1,1,1], [0,1,0]])
    elif conn == 8:
        conn = np.ones((3,3), dtype=int)
    b_erode = imerode(B, conn)
    p = B & ~b_erode
    return p
#----------------------------------------------------------------
def rgb2hsv(I):
    """
        RGB a HSV
        Uso:Conversión de una imagen en modelo “RGB” a modelo de color “HSV”.
        La sintaxis para su uso es:
            rgb2hsv(I)
        Parámetros:
            I--es la imagen en modelo RGB.
        Ejemplo:
            1. Se lee una imagen llamada “herramientas.jpg” y se almacena en una variable “A”:
                A=imread('herramientas.jpg')
            2. Se cambia la imagen al modelo de color “hsv” y se almacena en una variable “hsv”.
                hsv=rgb2hsv(A)
            3. Se muestra la imagen con la instrucción imshow.
                imshow(HSV)
    """
    [N,M,L] = I.shape
    if L != 3:
        print("Error rgb2hsv: La imagen debe ser de MxNx3")
    r = I[...,0]/255.0
    g = I[...,1]/255.0
    b = I[...,2]/255.0
    H, S, V = np.zeros((N, M), np.float32), \
        np.zeros((N, M), np.float32),  np.zeros((N, M), np.float32)
    hsv = np.zeros([N,M,3], np.float32)
    for i in range(0, N):
        for j in range(0, M):
            maximo = max((r[i, j], g[i, j], b[i, j]))
            minimo = min((r[i, j], g[i, j], b[i, j]))
            #d[i,j] = maximo-minimo
    
            if max == min: 
                H[i,j] = 0
            elif maximo == r[i,j]:
                H[i,j] = 60*np.mod((g[i,j]-b[i,j])/(maximo-minimo), 6)
            elif maximo == g[i,j]:
                H[i,j] = 60*((b[i,j]-r[i,j])/(maximo-minimo) +2)                    
            elif maximo == b[i,j]:
                H[i,j] = 60*((r[i,j]- g[i,j])/(maximo-minimo)+4)
            H[i, j] = (H[i, j] *255)/ 360

            if maximo == 0:
                S[i,j] = 0
            else: 
                S[i,j] = ((maximo-minimo)/maximo*255)
    
            V[i,j] = (maximo*255)
    hsv[...,0] = H
    hsv[...,1] = S
    hsv[...,2] = V
    return hsv/255.0
#---------------------------------------------------------------------------------------------------
def hsv2rgb(hsv):
    """
        HSV a RGB.
        Uso:Conversión de modelo “HSV” a modelo de color “RGB”.
        La sintaxis para su uso es:
            hsv2rgb(hsv)
        Parámetros:
            hsv--es la imagen en modelo de color HSV.
        Ejemplo:
            1. Se lee una imagen llamada “herramientas.jpg” y se almacena en una variable “A”:
                A=imread('herramientas.jpg')
            2. Se cambia la imagen al modelo “hsv”, se almacena en la variable “HSV” y se 
                muestra con la instrucción imshow.
                HSV=rgb2hsv(A)
                imshow(HSV)
            3. Se cambia la imagen de nuevo al modelo RGB, se almacena en una variable “RGB” y se 
                muestra con la instrucción imshow.
                RGB=hsv2rgb(HSV)
                imshow(RGB)
    """
    [N,M,L]= hsv.shape
    if (L != 3):
        print ("Error hsv2rgb: La imagen debe ser de MxNx3")
    F=1
    C = np.zeros((N,M), dtype=float)
    X = np.zeros((N,M), dtype=float)
    m = np.zeros((N,M), dtype=float)
    R = np.zeros((N,M), dtype=float)
    G = np.zeros((N,M), dtype=float)
    B = np.zeros((N,M), dtype=float)
    v = hsv[...,2]
    s = hsv[...,1]
    h = hsv[...,0]*360
    for i in range(0, N):
        for j in range(0, M):
            C[i,j] = v[i,j]*s[i,j]
            X[i,j] = C[i,j]*(1-np.abs(((h[i,j]/60)%2)-1))
            m[i,j] = v[i,j]-C[i,j]
            if(h[i,j] >= 0) and (h[i,j] < 60/F):
                R[i,j] = C[i,j]
                G[i,j] = X[i,j]
                B[i,j] = 0 
            elif (h[i,j] >= 60/F) and (h[i,j] < 120/F):
                R[i,j] = X[i,j]
                G[i,j] = C[i,j]
                B[i,j] = 0
            elif (h[i,j] >= 120/F) and (h[i,j] < 180/F):
                R[i,j] = 0
                G[i,j] = C[i,j]
                B[i,j] = X[i,j]              
            elif (h[i,j] >= 180/F) and (h[i,j] < 240/F):     
                R[i,j] = 0
                G[i,j] = X[i,j]
                B[i,j] = C[i,j]  
            elif (h[i,j] >= 240/F) and (h[i,j] < 300/F):
                R[i,j] = X[i,j]
                G[i,j] = 0
                B[i,j] = C[i,j]
            elif (h[i,j] >= 300/F) and (h[i,j] < 360/F):
                R[i,j] = C[i,j]
                G[i,j] = 0
                B[i,j] = X[i,j]
    r = (R+m)*255
    g = (G+m)*255
    b = (B+m)*255 
    rgb = np.zeros((N,M,3), dtype=float)
    rgb[...,0] = r
    rgb[...,1] = g
    rgb[...,2] = b
    return rgb/255
#------------------------------------------------------------------------------------------------