# Promoting-collective-intelligence

## Introduction

This custom code performs the simulations for the paper: **Promoting collective intelligence: The advantage of temporal star-structures**, by *Zhenglong Tian*, *Yao Meng*, *Wenxuan Fang* and *Aming Li*.  
The code applies to the simulations in **Fig 2** and **Fig 5** of the main text. 

## Included Files

### Monte Carlo Simulations
- `main.py`: Main script to run simulations for static and temporal networks.  
- `config.py`: Configuration parameters (network size, evolution rounds, payoff matrices, etc.).  
- `Create_snapshot.py`: Generates structural-temporal network snapshots (e.g., random, single-star, multi-cluster).  
- `EvolutionGame.py`: Implements evolutionary game dynamics and strategy updates.  
- `networkt.py`: Network generation utilities (e.g., static model scale-free networks).  
- `QueueAndStack.py`: Data structures for BFS/DFS traversal during snapshot creation.  
- `players.py` : Defines player classes and strategy initialization (referenced in `EvolutionGame.py`).  

### Supplementary

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

   The script automatically executes evolutionary games on generated snapshots. Results are saved to the `result` directory as `.txt` files containing cooperation frequencies.  

## License  
MIT License. 