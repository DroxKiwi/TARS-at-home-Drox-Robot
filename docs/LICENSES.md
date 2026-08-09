# Licences des composants (usage commercial potentiel)

> Ceci est une **checklist technique**, pas un avis juridique. Fais valider avant vente.

Le code de ce dépôt est sous **MIT** ([LICENSE](../LICENSE)).

## Stack retenue (permissive)

| Composant | Rôle | Licence | OK vente ? |
|-----------|------|---------|------------|
| Ce dépôt (TARS Voice) | Orchestre + UI | **MIT** | Oui |
| FastAPI / Uvicorn | HTTP + WS | MIT | Oui |
| **faster-whisper** | STT | **MIT** | Oui |
| OpenAI Whisper weights | Modèle STT | MIT | Oui |
| **Ollama** (runtime) | Serveur LLM | **MIT** | Oui |
| **kokoro-onnx** | Runtime TTS | MIT-friendly | Oui |
| **Kokoro-82M** weights | Modèle TTS | **Apache-2.0** | Oui (+ notice) |
| **Fun-CosyVoice 3** (option) | TTS clonage | **Apache-2.0** | Oui (+ notice) ; service `tars-tts-cosy` |
| NVIDIA CUDA base image | Conteneur | NVIDIA CUDA EULA | Runtime OK ; lire EULA redistrib |

## Modèles Ollama — à choisir toi-même

Ollama est MIT, **mais chaque modèle a sa propre licence**. Pour un produit vendable, privilégier :

| Exemples | Licence typique | Notes |
|----------|-----------------|--------|
| Qwen2.5 / Qwen3 (instruct) | Apache-2.0 | Bon choix commercial |
| Mistral / Mixtral | Apache-2.0 | OK |
| Llama 3.x | Llama Community License | Restrictions Meta — lire avant vente |
| Gemma | Gemma Terms | Restrictions Google |

Par défaut le `.env.example` pointe vers `qwen2.5:7b-instruct-q4_K_M` (Apache-2.0).

## Explicitement écarté (pour ce projet)

| Composant | Pourquoi |
|-----------|----------|
| **Piper TTS** (piper-tts actuel) | **GPL-3.0** — copyleft, problématique dans une image produit |
| Coqui XTTS v2 | CPML non-commercial |
| F5-TTS | CC-BY-NC |
| PersonaPlex / Moshi | OK commercial mais hors stack actuelle (qualité/langues) |
| Nemotron VoiceChat-11B | VRAM + card parfois « research only » |

## Notices à conserver dans un produit

1. Fichier `LICENSE` MIT de ce dépôt  
2. Attribution **Apache-2.0** Kokoro  
3. Attribution **faster-whisper** / Whisper  
4. Licence du **modèle Ollama** choisi  
5. Notices NVIDIA si tu redistribues l’image CUDA  

## Mise à jour

Si tu ajoutes un modèle/lib, documente-le ici **avant** de le merger.
