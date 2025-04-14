# Promoting-collective-intelligence

## Introduction

This custom code performs the simulations for the paper: **Promoting collective intelligence: The advantage of temporal star-structures**, by *Zhenglong Tian*, *Yao Meng*, *Wenxuan Fang* and *Aming Li*.  
The code applies to the simulations in **Fig 2** and **Fig 5** of the main text. 

## Included Files

### Monte Carlo Simulations
- `main.py`: Main script to run simulations for static and temporal networks.  
- `config.py`: Configuration parameters (network size, evolution rounds, payoff matrices, etc.).  
- `Create_snapshot.py`: Generates structural-temporal network snapshots (e.g., random, single-star, multi-cluster). Output files follow the format:

  ```bash
  [GraphType]_[N]N_[Params]_[f]f_[SubNet].npy
  ```

  Example: `SF_400N_4m_0.3f_Single-star.npy` represents:

  - `SF`: Barabási-Albert scale-free network.
  - `400N`: 400 nodes.
  - `4m`: Initial attachment parameter `m=4` (for scale-free networks).
  - `0.3f`: 30% of edges activated per snapshot.
  - `Single-star`: Structural type.

  Each `.npy` file contains a **3D numpy array** of shape `(snapshotNum, N, N)`, where:

  - `snapshotNum`: Number of snapshots (default 200).
  - `N`: Number of nodes (configured in `config.py`).
  - Each `N x N` matrix is a binary adjacency matrix for a snapshot.
- `EvolutionGame.py`: Implements evolutionary game dynamics and strategy updates.  
- `networkt.py`: Network generation utilities (e.g., static model scale-free networks).  
- `QueueAndStack.py`: Data structures for BFS/DFS traversal during snapshot creation.  
- `players.py` : Defines player classes and strategy initialization (referenced in `EvolutionGame.py`).  

### Others

- `README.md`: This documentation.  

## Dependencies

- **Python 3.10+**  
## Running the Software
1. **Generate Snapshots**:  
   Run `Create_snapshot.py` to create temporal network snapshots and save them to the `snapshot` directory:  

   ```bash
   python Create_snapshot.py
   ```
   *Note*: Adjust parameters in `config.py` (e.g., `N`, `snapshotNum`, `subNet`) for custom simulations.  

2. **Evolutionary Dynamics**:  

   Run `main.py` to simulate:  

   ```bash
   python main.py
   ```

   The script automatically executes evolutionary games on generated snapshots. Results are saved to the `result` directory as `.txt` files containing cooperation frequencies. Each `.txt` file contains **two columns of numerical data** representing:

   1. **Temptation parameter `b`**: Values from `blist` in `config.py` (e.g., `1.0, 1.1, ..., 2.5`).
   2. **Cooperation frequency `fc`**: Average fraction of cooperators across `EG_Rounds` (default 100).

   **Example**:

   ```
   1.0 0.9832  
   1.1 0.9547  
   ...  
   2.5 0.0121  
   ```

## License  
MIT License. 