# Subject announcement — Real-time speech recognition

## Français

**Reconnaissance vocale en temps réel : précision, latence et robustesse**

L'objectif de ce projet est de construire un système de reconnaissance automatique de la parole (ASR) en temps réel, dans l'esprit d'un assistant vocal : on appuie sur un bouton, on parle, et la transcription s'affiche. Le projet suit l'évolution du domaine. On part d'un système classique, dont on implémente soi-même les briques essentielles : descripteurs acoustiques MFCC, modèles de Markov cachés à mélanges de gaussiennes (GMM-HMM), modèles de langage n-grammes et décodage de Viterbi. On passe ensuite aux modèles neuronaux modernes (wav2vec 2.0, Whisper, modèles en flux de type transducteur), complétés par des modèles de langage externes. Il ne s'agit pas d'entraîner de grands modèles, mais de comparer des modèles pré-entraînés disponibles sur Hugging Face, de les combiner et de les configurer.

Chaque système est évalué sous deux angles. Objectivement, on mesure le taux d'erreur sur les mots (WER), la latence ressentie par l'utilisateur, le coût de calcul et la robustesse aux dégradations acoustiques (bruit, réverbération, canal téléphonique). Ces mesures sont représentées sous forme de fronts de Pareto, qui montrent ce que coûte chaque gain de précision. Qualitativement, on caractérise les compromis propres à chaque point du front : un modèle rapide qui laisse échapper des mots, un modèle précis mais trop lent pour une conversation, un modèle de langage qui produit des phrases fluides mais fausses. Le projet se conclut par une démonstration en direct, où chaque groupe branche ses systèmes à côté d'un système de référence embarqué. L'évaluation repose principalement sur la soutenance orale, où chaque groupe défend ses choix scientifiques. Aucune connaissance préalable en traitement de la parole n'est requise ; une aisance en Python et des bases en probabilités suffisent.

## English

**Real-time speech recognition: accuracy, latency and robustness**

The goal of this project is to build a real-time automatic speech recognition (ASR) system in the spirit of a voice assistant: press a button, speak, and the transcript appears. The project follows the history of the field. We start from a classical system whose core components you implement yourselves: MFCC acoustic features, hidden Markov models with Gaussian mixtures (GMM-HMM), n-gram language models and Viterbi decoding. We then move to modern neural models (wav2vec 2.0, Whisper, streaming transducers), complemented by external language models. The aim is not to train large models but to compare, combine and configure pre-trained models available on Hugging Face.

Every system is evaluated from two angles. Objectively, we measure word error rate (WER), the latency felt by the user, compute cost and robustness to acoustic degradations (noise, reverberation, telephone channel). These measurements are drawn as Pareto fronts, which show what each gain in accuracy costs. Qualitatively, we characterise the trade-offs of each point on the front: a fast model that drops words, an accurate model too slow for conversation, a language model that produces fluent but wrong sentences. The project ends with a live demo in which each group plugs its systems in next to an on-device reference system. Assessment rests mainly on the oral defence, where each group defends its scientific choices. No prior knowledge of speech processing is required; fluency in Python and basic probability are enough.

**Practical:** groups of 4 (3 possible) · bring a laptop with a microphone (a headset is fine) on which you can install Python packages.
