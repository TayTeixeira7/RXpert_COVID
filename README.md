## **RXpert COVID**
Trabalho de Processamento de Imagens

### **Equipe**
Taynara Garcia Teixeira

### **Descrição dos descritores implementados**
 **Hu Moments:** 
    Descritor usado para extrair características geométricas invariantes de forma, como posição, escala e rotação de imagens, permitindo descrever a forma dos objetos de acordo com a distribuição de intensidades dos pixels da imagem.
 **GLCM (Gray Level Co-occurrence Matrix):** 
    Descritor utilizado para extrair características de textura de imagens com base na frequência com que pares de pixels com níveis de cinza próximos se distribuem na imagem. As principais características extraídas incluem:
    - Contraste: Mede a intensidade do contraste entre um pixel e seus vizinhos ao longo de toda a imagem. Valores mais altos indicam maior contraste local.
    - Correlação: Avalia o quanto um pixel está correlacionado com seus vizinhos ao longo da imagem. Valores mais altos indicam uma correlação mais forte.
    - Energia: Mede a uniformidade da distribuição dos níveis de cinza. Valores mais altos indicam uma textura mais uniforme.
    - Homogeneidade: Mede a proximidade da distribuição dos elementos na GLCM à diagonal da matriz. Valores mais altos indicam que os elementos estão mais próximos da diagonal, representando uma textura mais suave.

### **Repositório do projeto**

    https://github.com/TayTeixeira7/RXpert_COVID

### **Classificador e acurácia**

```
- **Hu Moments:**   
    MLP (Multilayer Perceptron): 80.35%
    RF (Random Forest): 89.28%
    SVM (Support Vector Machine): 85.71%

- **GLCM (Gray Level Co-occurrence Matrix):** 
    MLP (Multilayer Perceptron): 94.64%
    RF (Random Forest): 89.28%
    SVM (Support Vector Machine): 96.42%
```

### **Instruções de uso (opcional)**
1. Certifique-se de que todas as bibliotecas estão instaladas corretamente. Para usar o descritor de textura GLCM é preciso fazer o import da biblioteca scikit-image.
2. Para calcular a matriz de co-ocorrência GLCM da imagem é
necessário trocar a função 'greycomatrix' para 'graycomatrix', e a função 'greycoprops' que extrai propriedades de textura para 'graycoprops' respectivamente.
3. Antes de extrair as features é necessário conferir
 se as pastas features_labels/glcm/train, features_labels/glcm/test, features_labels/humoments/train e features_labels/humoments/test estão vazias.
4. Verifique se os arquivos CSV gerados pelos scripts de extração de características estão no formato esperado.
