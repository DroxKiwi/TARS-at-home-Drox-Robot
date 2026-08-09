# Guide d'installation — NVIDIA PersonaPlex

PersonaPlex est un modèle **speech-to-speech full-duplex** (~7B, architecture Moshi) qui permet de contrôler :

- la **voix** (presets audio `NATF*`, `NATM*`, `VARF*`, `VARM*`)
- la **persona** (prompt texte : rôle, ton, contexte)

Il tourne confortablement sur une GPU NVIDIA avec **~24 Go de VRAM**.

Sources officielles :

- Code : [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex)
- Poids : [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) (modèle *gated*)
- Page recherche : [ADLR PersonaPlex](https://research.nvidia.com/labs/adlr/personaplex/)

---

## Architecture cible (réseau local)

```text
┌─────────────────────────────┐         LAN          ┌──────────────────────┐
│  VM IA (Linux + GPU 24 Go)  │ ◄──────────────────► │  PC client           │
│                             │                      │  Navigateur HTTPS    │
│  PersonaPlex server :8998   │                      │  micro + haut-parleurs│
│  Web UI + WebSocket /api/chat│                     └──────────────────────┘
└─────────────────────────────┘
```

Tu installes et sers le modèle **sur la VM**. Depuis ton PC, tu ouvres l’UI web (micro/speakers du navigateur).

---

## Prérequis matériel & logiciel

| Élément | Recommandé |
|--------|------------|
| GPU | NVIDIA **≥ 24 Go VRAM** (RTX 3090 / 4090, A10, A40…) |
| Usage VRAM | ~20–24 Go en mode full GPU |
| OS | **Linux** (Ubuntu 22.04+ recommandé) |
| CUDA / drivers | Drivers récents, CUDA **12.4+** |
| Blackwell (RTX 50xx) | PyTorch avec index **cu130** (voir plus bas) |
| Disque | **~50 Go** libres (modèle ~20 Go + deps + cache) |
| RAM système | **32 Go+** recommandé |
| Réseau | Port **8998/tcp** accessible depuis le PC |

Si la VRAM est juste insuffisante : option `--cpu-offload` (plus de RAM système, latence potentiellement dégradée).

> **Windows** n’est pas le chemin supporté pour le serveur. Sur la VM IA, reste sous Linux. WSL2 peut marcher mais Docker/native Linux est plus fiable.

---

## Coexistence avec Ollama (installé hors container)

Si la VM a déjà **Ollama en natif** (systemd / binaire host) et que tu ne veux **pas** le polluer :

| Risque | Impact sur Ollama | Mitigation |
|--------|-------------------|------------|
| `pip install` / venv host | Peut mélanger CUDA/Python | **Ne pas** installer PersonaPlex en pip sur l’hôte → **Docker only** |
| Disque | Ollama et PersonaPlex partagent le même SSD | Isoler le cache Docker ; surveiller l’espace |
| VRAM GPU | Les deux se marchent dessus | **Un seul gros modèle GPU à la fois** |
| Ports | Conflit réseau | Ollama `11434` vs PersonaPlex `8998` → OK |

### Règle d’or

- **PersonaPlex = uniquement via Docker Compose** (image + volumes dédiés).
- **Aucun** `pip install moshi` / PyTorch sur le Python système de la VM.
- Ollama reste tel quel (service host inchangé).

### Disque (~100 Go libres)

Ordre de grandeur PersonaPlex en Docker :

| Élément | Taille approx. |
|---------|----------------|
| Image CUDA + deps | ~8–15 Go |
| Poids modèle HF (cache) | ~20 Go |
| Marge build / couches Docker | ~5–10 Go |
| **Total prudent** | **~40–50 Go** |

Avec **100 Go libres**, c’est jouable **si** tu ne télécharges pas en parallèle d’énormes modèles Ollama. Avant de lancer :

```bash
df -h /
du -sh ~/.ollama 2>/dev/null || du -sh /usr/share/ollama 2>/dev/null
docker system df
```

Conseils espace :

- Cache PersonaPlex dans un dossier dédié (ex. `/data/personaplex/.cache`), pas dans `$HOME` saturé.
- Après tests : `docker image prune` / ne pas laisser plusieurs builds orphelins.
- Ne pas cloner inutilement plusieurs copies du repo.

### VRAM (critique)

PersonaPlex en full GPU ≈ **20–24 Go**. Ollama avec un modèle chargé en VRAM en même temps → **OOM** ou swap GPU catastrophique.

Avant `docker compose up` :

```bash
# Option A : arrêter Ollama le temps de la session PersonaPlex
sudo systemctl stop ollama

# Option B : décharger les modèles Ollama sans stopper le service
# (API) curl http://localhost:11434/api/generate -d '{"model":"...","keep_alive":0}'
nvidia-smi   # mémoire utilisée proche de 0 MiB côté process modèles
```

Quand tu as fini PersonaPlex :

```bash
docker compose down
sudo systemctl start ollama   # si tu l’avais stoppé
```

### Réseau

- Ollama : `11434` (inchangé)
- PersonaPlex : `8998` (pas de conflit)

### Ce qu’on n’installe PAS sur l’hôte

Pour protéger Ollama, **ignore** la section « méthode manuelle (venv Python) » de ce guide, sauf besoin de debug isolé. Reste sur Docker.

---

## 13. Nettoyage après test PersonaPlex

À exécuter **sur la VM** pour retirer le test sans toucher à Ollama.

### Arrêt + suppression conteneurs / images PersonaPlex

```bash
cd /data/personaplex/personaplex 2>/dev/null || cd ~/personaplex 2>/dev/null || true

# Si le compose tourne encore
docker compose down --rmi local --volumes --remove-orphans 2>/dev/null || true

# Au cas où le conteneur a un autre nom
docker ps -a --format '{{.ID}} {{.Names}} {{.Image}}' | grep -i personaplex || true
docker ps -a --format '{{.ID}} {{.Names}} {{.Image}}' | grep -i moshi || true
```

### Libérer le disque (cache modèle ~20 Go + images CUDA)

```bash
# Cache Hugging Face du test (dans le repo)
rm -rf /data/personaplex/personaplex/.cache 2>/dev/null || true
rm -rf /data/personaplex/.cache 2>/dev/null || true

# Certificats SSL locaux éventuels
rm -rf ~/personaplex-ssl 2>/dev/null || true

# Images Docker orphelines / build inutiles
docker image prune -af
docker builder prune -af

# Optionnel : supprimer tout le dossier du clone
# ATTENTION : irréversible
# rm -rf /data/personaplex
```

### Remettre Ollama

```bash
sudo systemctl start ollama
systemctl is-active ollama
nvidia-smi
df -h /
docker ps
```

Vérifications attendues :

- plus de conteneur PersonaPlex
- Ollama `active`
- VRAM libre (hors processus Ollama éventuels)
- espace disque récupéré (~30–50 Go selon ce qui avait été DL)

---

## 1. Préparer Hugging Face (une fois)

1. Crée un compte [Hugging Face](https://huggingface.co/) si besoin.
2. Ouvre [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) et **accepte la licence** du modèle.
3. Crée un token d’accès : [Settings → Access Tokens](https://huggingface.co/settings/tokens) (lecture suffit).
4. Sur la VM :

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
# Pour le rendre permanent (exemple bash) :
echo 'export HF_TOKEN=hf_xxxxxxxxxxxxxxxx' >> ~/.bashrc
source ~/.bashrc
```

Ne committe **jamais** ce token dans le dépôt.

---

## 2. Préparer la VM IA

### 2.1 Vérifier le GPU

```bash
nvidia-smi
```

Tu dois voir ta GPU, le driver, et ~24 Go de mémoire.

### 2.2 Dépendances système

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y git python3.12 python3.12-venv libopus-dev build-essential pkg-config

# Fedora / RHEL
# sudo dnf install -y git python3.12 opus-devel gcc gcc-c++ make pkgconf-pkg-config
```

`libopus-dev` est **obligatoire** (codec audio Opus pour le streaming).

### 2.3 Installer Docker Engine (Ubuntu)

À exécuter **sur la VM Linux** (pas sur ton PC Windows).

```bash
# Paquets de base
sudo apt update
sudo apt install -y ca-certificates curl

# Clé & dépôt Docker officiel
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Ton user peut lancer docker sans sudo
sudo usermod -aG docker "$USER"
```

**Déconnecte-toi / reconnecte-toi** (SSH) pour que le groupe `docker` soit pris en compte, puis :

```bash
docker version
docker compose version
```

### 2.4 Installer NVIDIA Container Toolkit (GPU dans Docker)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Vérification GPU dans un conteneur :

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Si tu vois le même tableau que `nvidia-smi` hôte → Docker + GPU OK.

Référence : [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

---

## 3. Installation — méthode recommandée : Docker Compose

C’est le chemin le plus simple pour une VM de lab.

```bash
# Token HF (ne le mets pas dans git)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx

# Dossier dédié (évite de saturer le home / de mélanger avec Ollama)
mkdir -p /data/personaplex
cd /data/personaplex

git clone https://github.com/NVIDIA/personaplex.git
cd personaplex

# Token HF pour le conteneur (.env est lu par docker compose)
cat > .env <<EOF
HF_TOKEN=${HF_TOKEN}
NO_TORCH_COMPILE=1
EOF
chmod 600 .env

# Avant de lancer : libérer la GPU si Ollama a un modèle chargé
# sudo systemctl stop ollama
# nvidia-smi

docker compose up --build
```

Le compose officiel monte `./.cache` → poids HF isolés dans ce dossier PersonaPlex, **pas** dans l’install Ollama.

Si ton disque data est ailleurs, tu peux adapter le volume dans `docker-compose.yaml` :

```yaml
volumes:
  - /data/personaplex/.cache:/root/.cache
```

Le `docker-compose.yaml` officiel :

- construit l’image CUDA 12.4 + Python 3.12
- expose le port **8998**
- monte le cache Hugging Face dans `./.cache`
- réserve 1 GPU NVIDIA

Au premier lancement, le téléchargement du modèle peut prendre plusieurs minutes.

L’UI est disponible sur :

```text
https://<IP_DE_LA_VM>:8998
```

Le navigateur affichera un avertissement de certificat auto-signé : **Avancé → Continuer** (normal en lab).

Arrêt :

```bash
docker compose down
```

---

## 4. Installation — méthode manuelle (venv Python)

Utile pour debugger ou customiser.

```bash
git clone https://github.com/NVIDIA/personaplex.git
cd personaplex

python3.12 -m venv .venv
source .venv/bin/activate

# Blackwell (RTX 50xx) UNIQUEMENT — à faire AVANT pip install moshi :
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

pip install ./moshi

# Si VRAM limite :
# pip install accelerate
```

Lancer le serveur (écoute sur toutes les interfaces pour accès LAN) :

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
SSL_DIR="${HOME}/personaplex-ssl"
mkdir -p "$SSL_DIR"

python -m moshi.server \
  --host 0.0.0.0 \
  --port 8998 \
  --ssl "$SSL_DIR"
```

Avec offload CPU si besoin :

```bash
python -m moshi.server \
  --host 0.0.0.0 \
  --port 8998 \
  --ssl "$SSL_DIR" \
  --cpu-offload
```

Depuis ton PC : `https://<IP_DE_LA_VM>:8998`

Un script d’aide est fourni dans ce dépôt : [`scripts/start-personaplex.sh`](../scripts/start-personaplex.sh).

---

## 5. Accès depuis ton PC (LAN)

1. Sur la VM, note l’IP locale :

```bash
hostname -I
# exemple : 192.168.1.50
```

2. Ouvre le port si un firewall est actif :

```bash
# ufw (Ubuntu)
sudo ufw allow 8998/tcp
sudo ufw reload
```

3. Sur le PC, dans Chrome/Firefox/Edge :

```text
https://192.168.1.50:8998
```

4. Autorise le micro quand le navigateur le demande.
5. Choisis une **voix** et un **prompt persona** dans l’UI, puis parle.

### Pare-feu / routeur

- Les deux machines doivent être sur le **même LAN** (ou VPN type WireGuard).
- Pas besoin d’exposer 8998 sur Internet pour un usage maison.

---

## 6. Voix disponibles

Presets fournis avec le modèle :

| Famille | IDs |
|--------|-----|
| Natural female | `NATF0`, `NATF1`, `NATF2`, `NATF3` |
| Natural male | `NATM0`, `NATM1`, `NATM2`, `NATM3` |
| Variety female | `VARF0` … `VARF4` |
| Variety male | `VARM0` … `VARM4` |

Les voix *Natural* sonnent en général plus conversationnelles ; *Variety* plus diversifiées.

---

## 7. Prompts persona (exemples)

### Assistant

```text
You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.
```

### Service client (style officiel)

```text
You work for First Neuron Bank which is a bank and your name is Sanni Virtanen. Information: The customer's transaction for $1,200 at Home Depot was declined. Verify customer identity.
```

### Conversation libre

```text
You enjoy having a good conversation.
```

### Persona type TARS (à expérimenter)

PersonaPlex généralise hors distribution d’entraînement. Exemple de départ :

```text
You enjoy having a good conversation. You are TARS, a companion robot with dry humor and high honesty. Keep answers concise. Prefer dry wit over jokes. Speak English unless asked otherwise.
```

Les prompts système / réponses tool-like doivent rester **ASCII** autant que possible pour la qualité vocale (évite tirets longs, emojis, etc.).

---

## 8. Évaluation offline (sans micro)

Utile pour tester sans Web UI :

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx

python -m moshi.offline \
  --voice-prompt "NATF2.pt" \
  --input-wav "assets/test/input_assistant.wav" \
  --seed 42424242 \
  --output-wav "output.wav" \
  --output-text "output.json"
```

Exemple service avec prompt fichier :

```bash
python -m moshi.offline \
  --voice-prompt "NATM1.pt" \
  --text-prompt "$(cat assets/test/prompt_service.txt)" \
  --input-wav "assets/test/input_service.wav" \
  --seed 42424242 \
  --output-wav "output.wav" \
  --output-text "output.json"
```

---

## 9. Options serveur utiles

| Option | Défaut | Rôle |
|--------|--------|------|
| `--host` | `localhost` | Mettre `0.0.0.0` pour accès LAN |
| `--port` | `8998` | Port HTTP(S) / WebSocket |
| `--ssl` | — | Dossier certificats (auto-générés si absents) |
| `--cpu-offload` | off | Couches sur CPU si VRAM insuffisante |
| `--device` | `cuda` | `cuda` ou `cpu` |
| `--hf-repo` | `nvidia/personaplex-7b-v1` | Repo des poids |

Endpoint WebSocket (utilisé par l’UI) : `/api/chat` sur le même port.

---

## 10. Dépannage

| Symptôme | Piste |
|----------|--------|
| OOM CUDA au chargement | `--cpu-offload` ; fermer les autres jobs GPU ; vérifier `nvidia-smi` |
| `HF_TOKEN` / 401 / gated | Accepter la licence HF + token valide |
| Navigateur bloque HTTPS | Certificat auto-signé → continuer quand même |
| Pas de son / micro | Permissions navigateur ; tester en HTTP local d’abord ; casque USB |
| Port inaccessible depuis le PC | `--host 0.0.0.0`, firewall 8998, même sous-réseau |
| Blackwell / RTX 50xx | Installer PyTorch `cu130` **avant** `pip install ./moshi` |
| Latence élevée avec offload | Normal : préférer full GPU 24 Go |
| Premier boot très long | Téléchargement modèle (~20 Go) + compilation éventuelle |

Variable utile si problèmes de compile Torch :

```bash
export NO_TORCH_COMPILE=1
```

(déjà présente dans le `docker-compose` officiel)

---

## 11. Pourquoi pas NemotronLabs VoiceChat-11B ici ?

Le modèle [NVIDIA-NemotronLabs-VoiceChat-11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) est très intéressant (full-duplex + tool calling), mais NVIDIA documente **≥ 80 Go de VRAM** (~66 Go runtime). Le checkpoint seul fait déjà ~44 Go. Sur une machine **24 Go**, PersonaPlex est le choix réaliste.

---

## 12. Checklist rapide

- [ ] `nvidia-smi` OK (24 Go)
- [ ] Licence HF acceptée + `HF_TOKEN`
- [ ] `libopus-dev` installé (ou Docker)
- [ ] Serveur up sur `0.0.0.0:8998`
- [ ] Firewall LAN ouvert
- [ ] PC : `https://<IP_VM>:8998` + micro autorisé
- [ ] Voix + prompt persona choisis
- [ ] Conversation test (interruption / backchannel)

---

## Licences

- Code PersonaPlex / Moshi (fork) : MIT
- Poids PersonaPlex : NVIDIA Open Model License
- Moshi upstream (Kyutai) : CC-BY-4.0
