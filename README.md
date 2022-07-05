# Transferência de Timbre na Síntese de Música
# Style transfer in Music Synthesis

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376L - Deep Learning aplicado a Síntese de Sinais*, 
oferecida no primeiro semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

 Integrantes do grupo.
 |Nome  | RA | Especialização|
 |--|--|--|
 | Lucas Hideki Ueda  | 156368  | Matemático Aplicado/MsC. Eng. Elétrica|
 | Leonardo B. de M. M. Marques  | 218479  | Eng. Eletricista|


## Resumo (Abstract)

> Resumo do objetivo, metodologia **e resultados** obtidos. Sugere-se máximo de 100 palavras. 

## Descrição do Problema/Motivação

 No domínio de imagem, a transferência de estilos artísticos, no sentido de texturas, já é uma realidade. Por exemplo, é possível transferir o estilo, definido pela forma dos traços e cores de uma pintura de Van Gogh ou Pablo Picasso, para uma imagem qualquer de entrada.

 Já no contexto de música, a propriedade de estilo se refere ao timbre do áudio em questão. Por exemplo, por mais que um saxofone e uma guitarra toquem a mesma nota, emitindo a mesma frequência fundamental, nós conseguimos identificar facilmente de ouvido qual instrumento está tocando, isso se deve ao timbre. O timbre é caracterizado tanto pelo envelope espectral, visualizado através de espectrogramas que diagnosticam a característica sonora completa do instrumento, como também pelo envelope temporal, que corresponde às chamadas características do envelope ADSR. 

 Em particular, modelos generativos necessitam de dados do que se quer gerar, no momento do treinamento. Então, caso quisessemos gerar um timbre musical diferente, seria necessário coletar, processar tais dados e re-treinar todo o modelo. Uma alternativa a isso, é a transferência de estilo one-shot, ou seja, transferir o timbre de um instrumento a partir de uma só amostra, que não necessariamente foi vista durante o treinamento pelo modelo.

## Objetivo

O objetivo do nosso projeto é explorar arquiteturais neurais generativas para realizar a transferência do timbre de referência (estilo) para o conteúdo musical desejado.

São objetivos específicos:

- Avaliar abordagens baseadas em normalizing flows e/ou variational autoencoders (VAEs) para a realização da transferência;
- Obter uma síntese controlada do sinal musical de sáida a partir do condicionamento do modelo generativo através de uma amostra do timbre de referência;

## Metodologia Proposta

Foi realizada uma replicação do trabalho [Self-Supervised VQ-VAE for One-Shot Music Style Transfer](https://arxiv.org/abs/2102.05749), publicado no ICASSP 2021. Este trabalho consiste numa abordagem baseada em um Autoencoder Variacional Quantizado (VQ-VAE) para transferência de timbre musical e tornou se então nosso modelo baseline. Julgou-se interessante essa abordagem devido ao fato do projeto ser de código aberto, disponibilizado em https://github.com/cifkao/ss-vq-vae; demonstrar bons resultados no artigo bem como na página de demonstrações; e possuir pelo menos um dos dois bancos de dados utilizados para treinamento também abertos. 

A partir daí, buscou-se introduzir um incremento no desempenho, através da proposta de um outro modelo generativo que apresenta excelentes resultados em outras áreas e ainda não explorado diretamente para a transferência de timbre: os [Normalizing Flows](https://proceedings.mlr.press/v37/rezende15.html). Especificamente, nós utilizamos o [Glow](https://proceedings.neurips.cc/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html) como tentativa de melhoria na tarefa de transferência de timbre musical. O Glow consiste em um modelo generativo baseado em normalizing flows e é composto pelas seguinters transformaçãoes: Actnorm, 1x1 Invertible Convlutions e uma Affine Coupling Layer. Nosso modelo glow é condicionado com as representações de conteúdo e estilo extraídas com dois encoders a partir de dois mel-espectrogramas de entrada, um contendo o conteúdo musical desejado e outro o timbre do instrumento. O condicionamento se dá através da introdução dessas representações nas camadas de Affine Coupling, através de uma camada Gated Tanh Unit. A arquitetura é mosotrada na figura abaixo.

![image](https://github.com/lucashueda/IA376L/blob/main/reports/figures/model.png)
![image](https://github.com/lucashueda/IA376L/blob/main/reports/figures/blocks.png)

Para treinar tanto o modelo basline como o proposto baseado em Glow, a base de dados utilizada e pré-processada para o treinamento foi a Lahk midi dataset, disponível em https://colinraffel.com/projects/lmd/, em sua versão completa, ou 'full', que contém 178k de arquivos MIDI, o que consiste em aproximadamente um ano de conteúdo musical na forma simbólica. 

Após finalização do treinamento, como medidas para analisar o desempenho, idealiza-se a implementação conjunta de: métricas objetivas, a fim de obter resultados preliminares e poder-se comparar dietamente com o modelo baseline VQ-VAE; uma avaliação perceptual, na qual os ouvintes julgariam qual a melhor transferência de conteúdo para o timbre desejado através de um teste AB, também comparando com o baseline; e por último julgar-se-ia a utilização do modelo como ferramenta criativa, intuito para o qual é desenvolvido, através da uso desse por artistas musicais em curtas composições, que evidenciariam um aumento da capacidade criativa de se compor música ao se utilizarem dessa nova ferramenta.

## Resultados e Discussão dos Resultados

> Na entrega parcial do projeto (E2), essa seção pode conter resultados parciais, explorações de implementações realizadas e 
> discussões sobre tais experimentos, incluindo decisões de mudança de trajetória ou descrição de novos experimentos, como resultado dessas explorações.

> Na entrega final do projeto (E3), essa seção deve elencar os **principais** resultados obtidos (não necessariamente todos), que melhor representam o cumprimento
> dos objetivos do projeto.

> A discussão dos resultados pode ser realizada em seção separada ou integrada à sessão de resultados. Isso é uma questão de estilo.
> Considera-se fundamental que a apresentação de resultados não sirva como um tratado que tem como único objetivo mostrar que "se trabalhou muito".
> O que se espera da sessão de resultados é que ela **apresente e discuta** somente os resultados mais **relevantes**, que mostre os **potenciais e/ou limitações** da metodologia, que destaquem aspectos
> de **performance** e que contenha conteúdo que possa ser classificado como **compartilhamento organizado, didático e reprodutível de conhecimento relevante para a comunidade**. 

O primeira passo foi processar a base de dados, transformandos os arquivos MIDI em arquivos wav. O primeiro passo consiste em rodar o arquivo prepare.ipynb no diretório https://github.com/cifkao/ss-vq-vae/tree/main/data/lmd/note_seq, em seguida, os arquivos wav's são gerados ao rodar o arquivos prepare.ipynb do diretório https://github.com/cifkao/ss-vq-vae/tree/main/data/lmd/audio_train. Apesar das instruções de ambiente virtual descritos no repositório, alguns problemas foram encontrados e solucionados:

- FluidSynth não tem instalador para windows e nem interface com o pyFluidSynth (Solução: Instalar manualmente e adicionar o bin do FluidSynth no pathing do windows)
- Também foi necessário instalar o SoX no windows pelo anaconda (Solução: rodar -> conda install -c conda-forge sox)

Após a adequação do ambiente o primeiro processamento foi iniciado, o processo total demorou cerca de 7 horas de processamento. O segundo processamento por sua vez levou cerca de 48 horas para terminar de rodar, gerando aproximadamente 82.6gb de audios no formato wav. Apesar de terminar o processamento, dado a grandeza dos dados alguns erros de espaço em disco impossibilitaram avanços na replicação do código. O google Colab não suporta facilmente tamanha quantidade de dados, e armazenar tamanha quantidade de dados no drive também se mostrou uma tarefa complicada. Além disso, na própria máquina local a compactação dos arquivos em um arquivo zip tem apresentado dificuldades, com estimativa de 7 horas de processamento (Imagem 1) e ocorrendo erros nas tentativas de finalizar a compactação.

![image](https://user-images.githubusercontent.com/19509614/170158786-0ce22b6c-a371-4e9d-8b7a-e40425901e63.png)

Imagem 1: Tempo estimado para compactar base processada.

Os próximos passos incluem rodar o modelo baseline em uma partição menor de dados, e averiguar a possibilidade de novas bases de dados, de menor volumetria e que possibilitem os experimentos utilizando drive e google colab.

## Conclusão

> A sessão de Conclusão deve ser uma sessão que recupera as principais informações já apresentadas no relatório e que aponta para trabalhos futuros.
> Na entrega parcial do projeto (E2) pode conter informações sobre quais etapas ou como o projeto será conduzido até a sua finalização.
> Na entrega final do projeto (E3) espera-se que a conclusão elenque, dentre outros aspectos, possibilidades de continuidade do projeto.

No atual estado do presente trabalho o cronograma inicialmente proposto pode ser mantido. Durante a etapa atual de familiarização e implantação de códigos já identificamos gargalos referente a base de dados, onde já estamos a par de novas bases e alternativas para nossa abordagem.

### Cronograma (a partir de 26/04/2022)
> - 1ª semana: Finalização da busca bibliográfica (definir conceitos e técnicas da literatura)
> - 2ª - 4ª semana (**etapa atual**): Familiarização e implantação baseado em códigos de referência. Definição e preparação da pipeline de experimentos de interesse.
> - 5ª - 7ª semana: Rodar experimentos de interesse e consolidar resultados parciais, métrica objetivas. Preparação de avaliação subjetiva.
> - 8ª semana: Avaliação subjetiva e finalização do paper final.
> - 9ª semana: Finalização e apresentação do trabalho.


## Referências Bibliográficas

 [Link para nossa tabela detalhada de referências.](https://docs.google.com/spreadsheets/d/1f9BfdOSueFFlQz8008JxTJVskYowop-zIzsyydMJlC8/edit?usp=drivesdk)
