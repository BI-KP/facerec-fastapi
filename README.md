<h3 align="center">Face Recognition Anti-Spoofing</h3>

---

<p align="center">Projek autentikasi menggunakan face recognition dengan fitur anti-spoofing
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Run Code](#running)
- [Model](#model)
- [Built Using](#built_using)

## About <a name = "about"></a>

Tujuan dari projek ini adalah memiliki sistem autentikasi yang robust dan aman

## Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites
Gunakan python versi 3.10.13

Buat virtual environment dengan menjalankan

```
python -m venv .venv
```

aktifkan venv dengan command\

windows:
```
./.venv/Scripts/activate
```
linux:

```
source .venv/bin/activate
```

### Dependencies
Install dengan menjalankan command berikut

```
pip install -r requirements.txt -c constraint.txt
```


## Run Code <a name = "running"></a>

Terdapat 2 versi kode yaitu dengan menggunakan vector database dan menggunakan pickle file.

Menjalankan versi pickle file:

```
uvicorn server:app --reload
```

Menjalankan versi vector database: 

```
uvicorn server-db:app --reload
```

## Model dan Detector <a name="model"></a>

Kita dapat menggunakan beberapa model dan detector face sesuai kebutuhan.\
Detector:
- MtCNN (untuk kebutuhan deteksi multiple face. Robust tetapi agak lambat)
- FastMtCNN (versi sederhana dari MtCNN, eksekusi jauh lebih cepat tetapi lebih cocok untuk single face)👍
- OpenCV (Versi umum, perlu konfigurasi lebih lanjut)
- RetinaFace (Detector face terbaik tetapi memliki waktu eksekusi yang lambat)\

Model:
- Facenet (500+ MB)
- Facenet512 (90+ MB)👍
- VGG-Face (500+ MB) 

## Built Using <a name = "built_using"></a>

- [FastAPI](https://fastapi.tiangolo.com/) - Server Framework
- [Deepface](https://github.com/serengil/deepface) - Face Recognition Library
- [Streamlit](https://streamlit.io/) - Web Apps Framework
- [Weaviate](https://weaviate.io/) - Database
