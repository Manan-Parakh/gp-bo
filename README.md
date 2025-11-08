# GP-BO

This project is a small implementation of **Gaussian Process-based Bayesian Optimization** inspired by a research paper I studied. The goal was to understand how surrogate models like Gaussian Processes can be used to efficiently optimize expensive or unknown functions.

---

## 📘 Overview

The project demonstrates:

* Sampling an objective function using **Latin Hypercube Sampling**
* Building a **Gaussian Process (GP)** surrogate model
* Running **Bayesian Optimization** using the **Expected Improvement (EI)** strategy
* Visualizing convergence, uncertainty, and search progress

---

## 🧠 Motivation

While reading about ML applications in engineering design, I wanted to practically see how Bayesian Optimization works for functions where evaluation is costly — similar to simulations in rocket engine design or CFD.
This small project helped me visualize how surrogate modeling balances exploration and exploitation.

---

## ⚙️ How to Run

```bash
git clone https://github.com/Manan-Parakh/gp-bo.git
cd gp-bo
pip install -r requirements.txt
python gp_bo_rocket_demo.py
```

---

## 📊 Output

The script:

* Generates a synthetic test function (“thrust”)
* Fits a GP surrogate model
* Iteratively finds the optimum using Bayesian Optimization
* Plots and saves results + logs in a CSV file

---

## 🧩 Dependencies

* `numpy`
* `scipy`
* `matplotlib`
* `scikit-learn`
* `pandas`
* `pyDOE` (for Latin Hypercube Sampling)

---

## 📝 Notes

This was a **learning project** — not meant for production — but can serve as a base for experimenting with other acquisition functions, kernels, or real-world design problems.

