# Transferência de timbre na síntese de música
# Style transfer in music synthesis

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

O objetivo do nosso projeto então será explorar arquiteturas autoencoder, onde o decoder condicionará o processo generativo ao estilo desejado. E no nosso caso o dado de entrada desse processo autoencoder é o espectrograma do sinal musical.

São objetivos específicos:

- A síntese controlada do sinal musical de sáida a partir de uma amostra do timbre de referência;
- Explorar o uso de abordagens generativas, tais como flows ou diffusion models.

## Metodologia Proposta

> Descrever de maneira clara e objetiva, citando referências, a metodologia proposta (E2) ou adotada (E3) para se alcançar os objetivos do projeto.
> Descrever bases de dados utilizadas.
> Citar algoritmos de referência.
> Justificar os porquês dos métodos escolhidos.
> Apontar ferramentas relevantes.
> Descrever metodologia de avaliação (como se avalia se os objetivos foram cumpridos ou não?).

## Resultados e Discussão dos Resultados

> Na entrega parcial do projeto (E2), essa seção pode conter resultados parciais, explorações de implementações realizadas e 
> discussões sobre tais experimentos, incluindo decisões de mudança de trajetória ou descrição de novos experimentos, como resultado dessas explorações.

> Na entrega final do projeto (E3), essa seção deve elencar os **principais** resultados obtidos (não necessariamente todos), que melhor representam o cumprimento
> dos objetivos do projeto.

> A discussão dos resultados pode ser realizada em seção separada ou integrada à sessão de resultados. Isso é uma questão de estilo.
> Considera-se fundamental que a apresentação de resultados não sirva como um tratado que tem como único objetivo mostrar que "se trabalhou muito".
> O que se espera da sessão de resultados é que ela **apresente e discuta** somente os resultados mais **relevantes**, que mostre os **potenciais e/ou limitações** da metodologia, que destaquem aspectos
> de **performance** e que contenha conteúdo que possa ser classificado como **compartilhamento organizado, didático e reprodutível de conhecimento relevante para a comunidade**. 

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

 [Self-Supervised VQ-VAE for One-Shot Music Style Transfer](https://arxiv.org/abs/2102.05749)
 
 [Groove2Groove: One-Shot Music Style Transfer with Supervision from Synthetic Data](https://hal.archives-ouvertes.fr/hal-02923548/document)
 
 [Text-to-Speech Synthesis Techniques for MIDI-to-Audio Synthesis](https://arxiv.org/pdf/2104.12292.pdf)

 [Music Style Transfer: A Position Paper](https://arxiv.org/pdf/1803.06841.pdf)
