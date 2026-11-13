"""Small-account Solana cost model (all values documented in README)."""
import numpy as np
PCT_COST = np.array([0.0075, 0.0175, 0.030])  # per side: [established pools, new launch pools, pump.fun bonding curve]
#   established: 0.25% LP fee + ~0.5% slippage/MEV;  launches: 0.25% fee + ~1.5% slippage;  curve: 1% pump.fun fee + 2% slippage
BASE_FEE_SOL = 0.000005          # signature fee per tx
PRIORITY_FEE_SOL = 0.0001        # typical priority fee per swap (busy periods are higher)
FIXED_SOL = BASE_FEE_SOL + PRIORITY_FEE_SOL   # charged per swap (2 per round trip)
RENT_SOL = 0.00203928            # SPL token account rent, locked while held
RENT_UNRECOVERED = 0.25          # share of rent never reclaimed (accounts left open / dust); rest returns on close
IMPACT_K = 0.5                   # extra slippage = 0.5 * size / entry-bar USD volume
FILL_CAP = 0.05                  # skip trade if size > 5% of entry-bar USD volume
BALANCES = np.array(list(range(5, 255, 5)) + [280], dtype=np.float64)   # $5..$250 step 5, plus $280 preset
