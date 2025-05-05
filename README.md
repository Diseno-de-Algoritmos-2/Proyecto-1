# CVRP Solver: Custom Algorithm vs. Paper Algorithm Comparison

This project implements and compares two algorithms for solving the **Capacitated Vehicle Routing Problem (CVRP)**:

* `Own/originalSolution.py`: A custom implementation.
* `Improved_Paper/main.py`: An implementation based on an algorithm from the literature.

It includes tools to run both algorithms individually and to perform automated comparison tests between them.

---

## 📦 Prerequisites

* **Python 3.8+** must be installed on your system.

---

## ⚙️ Installation

Follow these steps to set up your environment:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Diseno-de-Algoritmos-2/Proyecto-1
   cd Proyecto-1
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:

   * On Linux/macOS:

     ```bash
     source venv/bin/activate
     ```
   * On Windows:

     ```bash
     .\venv\Scripts\activate
     ```

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Algorithms Individually

Make sure your virtual environment is activated before executing the following commands.

Each algorithm runs the test case defined in `problemInstance.py`. You are free to modify this file to use your own instance.

### ▶️ Custom Algorithm

```bash
python Own/originalSolution.py
```

### ▶️ Paper Algorithm

```bash
python Improved_Paper/main.py
```

---

## 📊 Running Comparison Tests

To compare both algorithms automatically:

### ▶️ Run a Single Comparison Test

This command runs a single test case with a random graph size between 5 and 215:

```bash
python Comparación/main.py {TestNumber}
```

Replace `{TestNumber}` with any identifier (e.g., `1`, `42`, etc.).

### ▶️ Run Many Tests Automatically

To run a large number of comparison tests (default: 500), execute the provided Bash script from the root of the project:

```bash
bash Comparación/run_simulations.sh
```

This script will automatically run multiple randomized instances for comparison, by default it will run 500 instances.