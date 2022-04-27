# Transferência de timbre na síntese de música
# Style transfer in music synthesis

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376L - Deep Learning aplicado a Síntese de Sinais*, 
oferecida no primeiro semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

> Integrantes do grupo.
> |Nome  | RA | Especialização|
> |--|--|--|
> | Lucas Hideki Ueda  | 156368  | Matemático Aplicado/MsC. Eng. Elétrica|
> | Leonardo B. de M. M. Marques  | 218479  | Eng. Eletricista|


## Descrição Resumida do Projeto
> Descrição do tema do projeto, incluindo contexto gerador, motivação.
> Descrição do objetivo principal do projeto.
> Esclarecer qual será a saída do modelo de síntese, ou generativo.
> 
> Incluir nessa seção link para vídeo de apresentação da proposta do projeto (máximo 5 minutos).

## Metodologia Proposta

Dados disponíveis (Todos CC-BY 4.0):
>- MAESTRO dataset (pronto para download): https://magenta.tensorflow.org/datasets/maestro
>   - 200k MIDI files of piano recordings
>- Lakh MIDI dataset (pronto para download): https://colinraffel.com/projects/lmd/
>   - 170k MIDI files of mixed instruments

Lakh MIDI dataset parece ser o mais relevante para nossa tarefa por já se tratar de diversos instrumentos distintos. 

Abordagens de modelagem e artigo de referência:
> Identificamos que algumas técnicas já vistas em aula são utilizadas na literatura, entre elas as VAE's (e variações como o VQ-VAE) e as GAN's. Um trabalho bem em linha com o que queremos realizar é o "Self-Supervised VQ-VAE for One-Shot Music Style Transfer", onde as VQ-VAE's são utilizadas para modelar o estilo musical. Uma possível variação seria utilizar Normalizing Flows para a mesma tarefa.

Ferramentas a serem utilizadas:
> Em média a literatura reporta cerca de 20h de treinamento utilizando uma GPU V100. Acreditamos portanto que o uso do **colab (P100 ou T4)** e **máquina prória (RTX2060)**, permitirão treinamentos com mais tempo de treinamento. Além disso, utilizaremos o **google drive** para centralizar os scripts de experimentos e todos os arquivos resultantes (revisão bibliográfica, análise de resultados, resultados parciais, etc). O código será versionado e armazenado no **github** e o paper final escrito em LaTeX utiliznado o **overleaf**.

Resultados esperados:
> Esperamos uma rede generativa capaz de gerar áudios a partir de arquivos MIDI. Em particular, de forma que o estilo de tais áudios possam ser condicionados ao estilo de um arquivo MIDI de referência. Também esperamos entender mais o mecanismo de geração de áudios a partir de metodologias novas em redes generativas, como os Normalizing Flows ou os Denoising Diffusions.

Proposta de avaliação
> Se possível, uma avaliação perceptual, onde ouvintes deverão dizer se o estilo gerado pelo modelo (no timbre do áudio base) condiz com o estilo da referência.
> 
> Métricas objetivas:
> 
>     - Log-Spectral Distance: RMSE do DB-scale mel-spectrogram.
>     
>     - Pitch error: Distância Jaccard entre as curvas de pitch.
>     
>     - Timbre error: Dissimilarity score de uma rede treinada nos 1-13 componentes MFCC.

## Cronograma (a partir de 26/04/2022)
> - 1ª semana: Finalização da busca bibliográfica (definir conceitos e técnicas da literatura)
> - 2ª - 4ª semana: Familiarização e implantação baseado em códigos de referência. Definição e preparação da pipeline de experimentos de interesse.
> - 5ª - 7ª semana: Rodar experimentos de interesse e consolidar resultados parciais, métrica objetivas. Preparação de avaliação subjetiva.
> - 8ª semana: Avaliação subjetiva e finalização do paper final.
> - 9ª semana: Finalização e apresentação do trabalho.


## Referências Bibliográficas
> [Self-Supervised VQ-VAE for One-Shot Music Style Transfer](https://arxiv.org/abs/2102.05749)
> 
> [Groove2Groove: One-Shot Music Style Transfer with Supervision from Synthetic Data](https://hal.archives-ouvertes.fr/hal-02923548/document)
> 
> [Text-to-Speech Synthesis Techniques for MIDI-to-Audio Synthesis](https://arxiv.org/pdf/2104.12292.pdf)
>
> [Music Style Transfer: A Position Paper](https://arxiv.org/pdf/1803.06841.pdf)
