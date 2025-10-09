<div align="center">

# 🎨 FairImagen （Neurips2025）

### Post-Processing for Bias Mitigation in Text-to-Image Models

<p>
  <img src="https://img.shields.io/badge/Python-3.9-blue.svg" alt="Python 3.9"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
  <img src="https://img.shields.io/badge/NeurIPS-2025-red.svg" alt="NeurIPS 2025"/>
</p>

<p>
  <strong>Zihao Fu</strong><sup>1</sup> ·
  <strong>Ryan Brown</strong><sup>2</sup> ·
  <strong>Shun Shao</strong><sup>3</sup> ·
  <strong>Kai Rawal</strong><sup>4</sup> ·
  <strong>Eoin D. Delaney</strong><sup>5</sup> ·
  <strong>Chris Russell</strong><sup>4</sup>
</p>

<p>
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/93/CUHK_Emblem_%281973-2008%29.svg" height="16" alt="CUHK"/> <sup>1</sup>The Chinese University of Hong Kong ·
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/a2/Uni_Oxford_logo.svg" height="16" alt="Oxford"/> <sup>2</sup>University of Oxford ·
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Coat_of_Arms_of_the_University_of_Cambridge.svg" height="16" alt="Cambridge"/> <sup>3</sup>University of Cambridge<br/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/a2/Uni_Oxford_logo.svg" height="16" alt="OII"/> <sup>4</sup>University of Oxford ·
  <img src="https://upload.wikimedia.org/wikipedia/commons/7/7f/Trinity_College_Dublin_Arms.svg" height="16" alt="TCD"/> <sup>5</sup>Trinity College Dublin
</p>

<p>
  📧 <a href="mailto:zihaofu@cuhk.edu.hk">zihaofu@cuhk.edu.hk</a>
</p>

</div>

---

## 📸 Visual Comparison

<table>
  <tr>
    <td align="center">
      <img src="https://private-user-images.githubusercontent.com/1419566/498914070-c60e9f3b-c902-45ca-bbc6-257f599fc704.jpeg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTk5OTI1MTgsIm5iZiI6MTc1OTk5MjIxOCwicGF0aCI6Ii8xNDE5NTY2LzQ5ODkxNDA3MC1jNjBlOWYzYi1jOTAyLTQ1Y2EtYmJjNi0yNTdmNTk5ZmM3MDQuanBlZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMDklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDA5VDA2NDMzOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWRjY2JkNGVmN2RkNTZkZDI1NjcxYTQ0NGE5YjE0MDY3ZDQzNTVjNWE5NTM4ZThmODA3OTgyYjBkNTg5OTIwZGUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.FTC2ijnzaODtVYM3OZbmRwWSda3Be-1dbbBnhlRwmNU" alt="Baseline" width="400"/><br/>
      <b>Baseline (no debiasing)</b>
    </td>
    <td align="center">
      <img src="https://private-user-images.githubusercontent.com/1419566/498914068-de881000-fa25-4409-803b-f104ebe3c92b.jpeg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTk5OTI1MTgsIm5iZiI6MTc1OTk5MjIxOCwicGF0aCI6Ii8xNDE5NTY2LzQ5ODkxNDA2OC1kZTg4MTAwMC1mYTI1LTQ0MDktODAzYi1mMTA0ZWJlM2M5MmIuanBlZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMDklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDA5VDA2NDMzOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWFjOTFhYTcwYmVmZWIxNzBlNGJlNzBmOTk5NjYyMjAyNGMwNDU0MTk5NDNhYzRmMjk5OGNkOWMzNjhhZWIwZmQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.PgRtIGReX7gnKEpqI-bv6PeVRQW9SuCKM9SK0q1NTPY" alt="FairPCA" width="400"/><br/>
      <b>FairPCA debiasing (gender)</b>
    </td>
  </tr>
</table>

<p align="center"><i>These examples illustrate how FairPCA mitigates gender bias while preserving prompt fidelity.</i></p>

---

## 🌟 Overview

**FairImagen** is an open-source, **post-hoc debiasing framework** for text-to-image diffusion models. It operates on prompt embeddings to mitigate demographic bias (e.g., gender, race) **without retraining or modifying model weights**.

The method integrates **Fair Principal Component Analysis (FairPCA)** to project text embeddings into a subspace that minimizes group-specific information while preserving semantic content. We further employ **empirical noise injection** to balance fairness and fidelity, and introduce a **unified cross-demographic projection** to debias multiple attributes simultaneously.

---

## ✨ Highlights

<table>
  <tr>
    <td>🚀</td>
    <td><b>Training-free</b>, black-box compatible post-processing</td>
  </tr>
  <tr>
    <td>🎯</td>
    <td><b>FairPCA projection</b> removes group-dependent directions while preserving semantics</td>
  </tr>
  <tr>
    <td>⚖️</td>
    <td><b>Empirical noise injection</b> for controllable fairness–fidelity trade-offs</td>
  </tr>
  <tr>
    <td>🌐</td>
    <td><b>Cross-demographic debiasing</b> handles multiple protected attributes jointly</td>
  </tr>
  <tr>
    <td>🔧</td>
    <td><b>Encoder flexibility:</b> default, T5, and OpenCLIP</td>
  </tr>
</table>

---

## 📊 Comparison of Debiasing Paradigms

<table>
  <thead>
    <tr>
      <th>Criteria</th>
      <th align="center">Prompt-based</th>
      <th align="center">Fine-tuning-based</th>
      <th align="center">Post-hoc editing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Training-free</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
    </tr>
    <tr>
      <td>Black-box compatible</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
    </tr>
    <tr>
      <td>Low human effort</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
    </tr>
    <tr>
      <td>Low computational cost</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
    </tr>
    <tr>
      <td>Generalizable to new prompts</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
    </tr>
    <tr>
      <td>Strong bias mitigation</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
    </tr>
    <tr>
      <td>Preserves prompt fidelity</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
    </tr>
    <tr>
      <td>Easy deployment</td>
      <td align="center">❌</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
    </tr>
  </tbody>
</table>

---

## 🛠️ Installation

<details open>
<summary><b>Requirements</b></summary>

- Python 3.9
- CUDA-enabled GPU recommended

</details>

```bash
# Create and activate conda environment
conda create -n fairimagen python=3.9
conda activate fairimagen

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Data

- Prompt lists reside in `FairImagen/data/`
- **`test.json`** phrasing: `"Generate a photo of a face of a <role>"`
- **`dev.json`** contains 30 prompts with the same phrasing and does not overlap with `test.json`

---

## 🚀 Quick Start

<details open>
<summary><b>Baseline (no debiasing)</b></summary>

```bash
./scripts/run.sh "data=test,protect=[gender]" "proc=base"
```

</details>

<details open>
<summary><b>FairPCA debiasing (gender)</b></summary>

```bash
./scripts/run.sh "data=test,protect=[gender]" "proc=fpca,remove,enoise=0.6,hdim=1800"
```

</details>

<details open>
<summary><b>FairPCA with T5 encoder</b></summary>

```bash
./scripts/run.sh "data=test,protect=[gender]" "proc=fpca,remove,enoise=0.6,hdim=1800,encoder=T5"
```

</details>

<details open>
<summary><b>FairPCA with OpenCLIP encoder</b></summary>

```bash
./scripts/run.sh "data=test,protect=[gender]" "proc=fpca,remove,enoise=0.6,hdim=1800,encoder=OpenClip"
```

</details>

<details open>
<summary><b>Multi-attribute with cross interactions (gender and race)</b></summary>

```bash
./scripts/run.sh "data=test,protect=[gender,race]" "proc=fpca,remove,enoise=0.6,hdim=1400,cross"
```

</details>

---

## ⚙️ Parameters

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
      <th>Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>data</code></td>
      <td>Dataset name (uses <code>FairImagen/data/{name}.json</code>)</td>
      <td><code>test</code>, <code>dev</code></td>
    </tr>
    <tr>
      <td><code>protect</code></td>
      <td>Protected attributes to debias</td>
      <td><code>[gender]</code>, <code>[race]</code>, <code>[gender,race]</code></td>
    </tr>
    <tr>
      <td><code>proc</code></td>
      <td>Processor type</td>
      <td><code>base</code> (baseline), <code>fpca</code> (FairPCA)</td>
    </tr>
    <tr>
      <td><code>remove</code></td>
      <td>Enable debiasing when using <code>fpca</code></td>
      <td>Flag (no value)</td>
    </tr>
    <tr>
      <td><code>hdim</code></td>
      <td>Latent dimension for projection</td>
      <td><code>1800</code></td>
    </tr>
    <tr>
      <td><code>enoise</code></td>
      <td>Noise level to reinject after removal</td>
      <td><code>0.6</code></td>
    </tr>
    <tr>
      <td><code>encoder</code></td>
      <td>Text encoder override</td>
      <td><code>T5</code>, <code>OpenClip</code></td>
    </tr>
    <tr>
      <td><code>cross</code></td>
      <td>Consider attribute interactions in multi-attribute FPCA</td>
      <td>Flag (no value)</td>
    </tr>
  </tbody>
</table>

---

## 📁 Outputs

Generated images and experiment artifacts are saved under `output/` (subdirectories depend on parameters and runs).

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{fu2025fairimagen,
  title     = {FairImagen: Post-Processing for Bias Mitigation in Text-to-Image Models},
  author    = {Fu, Zihao and Brown, Ryan and Shao, Shun and Rawal, Kai and Delaney, Eoin D. and Russell, Chris},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

---

## 🙏 Acknowledgments

We gratefully acknowledge the [Amazon Science Fair PCA project](https://github.com/amazon-science/fair-pca) for inspiration and reference.

---

## 📄 License

<p align="center">
  <a href="LICENSE">MIT License</a>
</p>

<div align="center">

---

Made with ❤️ by the FairImagen Team

</div>
