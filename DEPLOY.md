# Deploying the web app

The app (`app.py`) is a Gradio web page. You can upload a component image, paste one, or take one with a webcam, and it shows **FAULTY** or **NOT FAULTY** with the defect type. A second tab checks many images at once and lets you download the results as a CSV.

| Option | Who can open it | How long it lasts | Your PC must stay on? | Cost |
|---|---|---|---|---|
| 1. Local | only you | while it runs | yes | free |
| 2. Temporary share link | anyone with the link | up to 1 week | yes | free |
| **3. Render** | anyone | permanent | **no** | **free** |
| 4. Hugging Face Spaces | anyone | permanent | no | Gradio Spaces need a PRO plan |

Commands for options 1 and 2 are run from the project folder:

```bash
cd "/home/vikas-more/Program Files/Aditya/fault_model"
```

## 1. Run locally

```bash
.venv/bin/python app.py
```

Then open http://127.0.0.1:7860. Press Ctrl+C in the terminal to stop it.

## 2. Temporary public link

```bash
.venv/bin/python app.py --share
```

The terminal prints a link like `https://xxxxxxxx.gradio.live`. Anyone with it can use the app, including on phones. The link stops working when you stop the app, or after one week at most.

## 3. Deploy on Render (free, permanent)

The repo is already set up for Render:
- `render.yaml` holds all the settings.
- `requirements-render.txt` is the CPU-only package list.
- `models/best_traced.pt` is a low-memory version of the model.

Tested in a Render-like setup, the app uses at most **about 440 MB of RAM**, under the free plan's 512 MB. Each image takes about **0.15 seconds**.

### Step 1: Push the project to GitHub

Render deploys straight from your GitHub repo:

```bash
cd "/home/vikas-more/Program Files/Aditya/fault_model"
git add .
git status        # check: models/best.pt, models/best_traced.pt, app.py, render.yaml... should be listed
                  #        .venv/, cache/ and the dataset should NOT be listed
git commit -m "Add web app and Render deployment"
git push origin main
```

### Step 2: Create a Render account

1. Go to https://render.com and click **Get Started**.
2. Choose **Sign up with GitHub**. This lets Render see your repositories.

### Step 3: Create the service from the Blueprint

1. In the Render dashboard, click **New +** → **Blueprint**.
2. Connect the repository **`vikas-more16/transmission-line-defect-classifier`**. If it isn't listed, click **Configure account** and give Render access to it.
3. Render reads `render.yaml` and shows one web service, **transmission-line-defect-classifier**, on the **Free** plan.
4. Click **Apply** (or **Deploy Blueprint**).

<details>
<summary>Alternative: set it up by hand instead of the Blueprint</summary>

**New +** → **Web Service** → pick the repo, then fill in:

| Field | Value |
|---|---|
| Language / Runtime | Python 3 |
| Branch | main |
| Build Command | `pip install -r requirements-render.txt` |
| Start Command | `python app.py` |
| Instance Type | **Free** |

Under **Environment variables**, add `PYTHON_VERSION` = `3.12.8`, `OMP_NUM_THREADS` = `1` and `MALLOC_ARENA_MAX` = `2`. Then click **Create Web Service**.
</details>

### Step 4: Wait for the first build (5–10 minutes)

Watch the **Logs** tab. The build installs PyTorch, which is the slow part. It's done when you see:

```
==> Your service is live 🎉
```

### Step 5: Open and share your app

The URL is shown at the top of the service page, for example:

```
https://transmission-line-defect-classifier.onrender.com
```

Anyone can open this link. No account is needed to use it.

### Things to know about Render's free plan

- **It sleeps after 15 minutes with no visitors.** The next visitor waits about **30–60 seconds** while it wakes up, and after that it's fast again. This is normal on the free plan.
- **Updates are automatic.** Every `git push` to `main` rebuilds and redeploys the app.
- **After retraining**, recreate the low-memory model before pushing:
  ```bash
  .venv/bin/python export_traced.py
  git add models/ && git commit -m "Update model" && git push
  ```

### If something goes wrong

| Problem in the Logs tab | Fix |
|---|---|
| `Out of memory` / `Ran out of memory (used over 512MB)` | Make sure `models/best_traced.pt` is in the repo (`git ls-files models`). Without it, the app loads the heavier checkpoint. |
| `No open ports detected` | The Start Command must be `python app.py` (the app reads Render's `PORT` automatically). |
| `FileNotFoundError: models/...` | The model files weren't pushed. Run `git add models/best.pt models/best_traced.pt`, then commit and push. |
| Build takes very long or times out | Check that the build command uses `requirements-render.txt`, not `requirements.txt`, which would pull the 3 GB GPU version of PyTorch. |

## 4. Hugging Face Spaces (paid)

Hugging Face now requires a PRO plan for Gradio Spaces. If you have one:

```bash
.venv/bin/hf auth login
.venv/bin/python deploy_space.py --push YOUR_USERNAME/transmission-line-defect-classifier
```

## Using the app from code

A running app, local or on Render, also works as an API:

```python
from gradio_client import Client, handle_file

client = Client("https://transmission-line-defect-classifier.onrender.com/")   # or "http://127.0.0.1:7860/"
verdict_html, probabilities = client.predict(handle_file("image.jpg"), api_name="/classify")
print(probabilities["label"])   # e.g. "Rust / corrosion" or "Good"
```
