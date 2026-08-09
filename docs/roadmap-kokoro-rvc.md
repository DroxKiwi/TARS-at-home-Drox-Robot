# Feature prévue — Kokoro → RVC (voix personnalisée basse latence)

> Statut : **roadmap**, non implémenté. CosyVoice reste l’option clonage zero-shot actuelle (plus lente).

## Objectif

Garder la latence de **Kokoro** tout en appliquant une **voix cible** (toi / TARS) via conversion, sans remplacer le TTS par un gros modèle de clonage à chaque phrase.

## Flux cible

```text
LLM → Kokoro (prosodie, rapide) → RVC (timbre) → haut-parleurs
```

- Kokoro : rythme / intonation (voix stock).
- RVC : change surtout le « grain » vers le modèle entraîné.

## Comment appliquer une voix (concret)

Ce n’est **pas** un sample zero-shot comme CosyVoice.

1. **Corpus** — ~5–10 min d’audio propre de la voix cible (mono, peu de bruit).
2. **Entraînement RVC** — une fois (local GPU ou Colab) → artefacts `.pth` (+ index éventuel).
3. **Déploiement** — service / étape `RVC` optionnelle après Kokoro, modèle monté depuis `data/voices/rvc/…`.
4. **UI** — choix Kokoro + profil RVC (ex. `tars`), pas de re-upload de prompt à chaque session.

Changer de voix = entraîner / charger un autre modèle RVC.

## Pourquoi pas CosyVoice pour ça

| | CosyVoice (actuel) | Kokoro → RVC (prévu) |
|---|---|---|
| Mise en place | Sample + transcript, immédiat | Train une fois |
| Latence | Élevée (RTF ~1) | Proche de Kokoro |
| VRAM / tour | Activations lourdes | Léger si RVC bien réglé |

## Suite technique (quand on s’y met)

- [ ] Service ou étape pipeline `kokoro → rvc`
- [ ] Volume `data/voices/rvc/` + doc d’entraînement minimale
- [ ] Toggle UI « RVC off / profil »
- [ ] Streaming par chunks (Kokoro puis RVC) pour la latence perçue
- [ ] Budget VRAM avec Whisper + Ollama sur 12 Go

## Références internes

- TTS actuel : `services/voice` (Kokoro) + `services/tts-cosy` (CosyVoice optionnel)
- Discussion latence Cosy : cache `add_zero_shot_spk` + fp16 déjà en place côté Cosy
