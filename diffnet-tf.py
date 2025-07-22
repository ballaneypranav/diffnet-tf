import numpy as np
import random, math, re, sys
import tensorflow as tf
import pandas as pd
import argparse

#
# Extended DiffNet implementation with TensorFlow
#
# Usage:
#  python diffnet-tf.py --input data.csv --epochs 1000 --nrounds 20 --lr 0.1
#


@tf.function
def cost(g, a, ar, s):
    """
    Calculates the cost function for the DiffNet model.
    This function is decorated with @tf.function to compile it into a high-performance
    TensorFlow graph.
    """
    cw = []
    # pattern is "rcpt2:lig2|rcpt1:lig1"
    pattern = re.compile('([^:\|]*):([^:\|]*)\|([^:\|]*):([^:\|]*)')
    for key, value in a.items(): 
        matches = pattern.search(key)
        rcpt1 = matches.group(1)
        lig1  = matches.group(2)
        rcpt2 = matches.group(3)
        lig2  = matches.group(4)
        av = a[key] + ar[key]
        if rcpt2 != "" and lig2 != "":
            if rcpt1 != rcpt2:
                if lig1 != lig2:
                    # Case D: Swapping transformation 
                    jb = rcpt2 + ":" + lig2 + "|:"
                    ib = rcpt2 + ":" + lig1 + "|:"
                    ja = rcpt1 + ":" + lig2 + "|:"
                    ia = rcpt1 + ":" + lig1 + "|:"
                    cw.append( ( -(g[jb]-g[ib])+(g[ja]-g[ia]) - av )/s[key] )
                else:
                    # Case C: Hopping transformation 
                    ib = rcpt2 + ":" + lig1 + "|:"
                    ia = rcpt1 + ":" + lig1 + "|:"
                    cw.append( ( (g[ib]-g[ia]) - av )/s[key] )
            else:
                # Case B: Relative Binding Free Energy (RBFE)
                ja = rcpt1 + ":" + lig2 + "|:"
                ia = rcpt1 + ":" + lig1 + "|:"
                cw.append( ( (g[ja]-g[ia]) - av )/s[key] )
        else:
            # Case A: Absolute Binding Free Energy (ABFE)
            ia = rcpt1 + ":" + lig1 + "|:"
            cw.append( ( g[ia] - av )/s[key] )
    c = tf.convert_to_tensor( cw )
    cost_val = tf.tensordot(c,c,axes=1)
    return cost_val

def main():
    """
    Main function to parse arguments, run optimization, and perform error analysis.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='Extended DiffNet implementation with TensorFlow.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=None,
        help='Input CSV file path.\nIf not provided, reads from standard input (stdin).'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1000,
        help='Number of optimization epochs for the main fit.'
    )
    parser.add_argument(
        '--nrounds',
        type=int,
        default=20,
        help='Number of rounds for error analysis.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help='Learning rate for the Adam optimizer.'
    )
    args = parser.parse_args()

    # --- Data Loading ---
    input_source = args.input if args.input is not None else sys.stdin
    try:
        #transid,DGb,DGberr,minsample,maxsample,optimize
        dgdata = pd.read_csv(input_source, header=0, delimiter=",", comment="#")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input}'", file=sys.stderr)
        sys.exit(1)
    
    print("--- Input Data ---")
    print(dgdata.to_string())
    print("------------------\n")

    # --- Data Processing ---
    pattern = re.compile('([^:\|]*):([^:\|]*)\|([^:\|]*):([^:\|]*)')

    g = {}
    variables = []
    # Separate absolute free energies (g) which are variables to be optimized
    for key,dgb,opt in zip(dgdata['transid'],dgdata['DGb'],dgdata['optimize']):
        matches = pattern.search(key)
        rcpt1, lig1, rcpt2, lig2 = matches.groups()
        if rcpt2 == "" and lig2 == "": # This identifies ABFEs
            g[key] = tf.Variable(dgb, dtype=tf.float64)
            if opt == 'T':
                variables.append(g[key])

    # Separate relative free energies (a) and their errors (s), which are constants
    a = {}
    s = {}
    for key,dgb,dgberr,opt in zip(dgdata['transid'],dgdata['DGb'],dgdata['DGberr'],dgdata['optimize']):
        if opt != 'T':
            a[key] = tf.constant(float(dgb), dtype=tf.float64)
            s[key] = tf.constant(float(dgberr), dtype=tf.float64)

    # Initialize artificial residuals for error analysis
    ar = { key:tf.constant( 0.0, dtype=tf.float64) for key in a }

    # --- Main Optimization ---
    optimizer = tf.keras.optimizers.Adam(args.lr)
    print("--- Starting Main Optimization ---")
    print(f"Epochs: {args.epochs}, Learning Rate: {args.lr}")
    for i in range(args.epochs):
        with tf.GradientTape() as tp:
            costf = cost(g, a, ar, s)
        gradients = tp.gradient(costf, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        if (i + 1) % 200 == 0 or i == args.epochs - 1:
             print(f"Epoch {i+1}/{args.epochs}, Cost: {costf.numpy():.4f}")
    print("--- Main Optimization Complete ---\n")

    gbest = { key:g[key].numpy() for key in g }
    g2 = { key:0.0 for key in g  }

    # --- Error Analysis ---
    print("--- Starting Error Analysis ---")
    print(f"Rounds: {args.nrounds}")
    for r_idx in range(args.nrounds):
        # Create a new set of random residuals for each round based on experimental error
        ar_err = { key:tf.random.normal([1],stddev=s[key], dtype=tf.float64)[0] for key in a }
        
        # Re-run optimization for this round of error analysis
        for _ in range(args.epochs):
            with tf.GradientTape() as tp:
                costf = cost(g, a, ar_err, s)
            gradients = tp.gradient(costf, variables)
            optimizer.apply_gradients(zip(gradients, variables))
        
        # Accumulate squared differences from the best-fit values
        gg = { key:g[key].numpy() for key in g }
        for key in g2:
            g2[key] += np.power(gg[key]-gbest[key], 2) / float(args.nrounds)
        print(f"Error analysis round {r_idx+1}/{args.nrounds} complete.")
    print("--- Error Analysis Complete ---\n")

    gerr = { key:np.sqrt(g2[key]) for key in g2  }

    # --- Final Results ---
    print("--- Final Results (value +- error) ---")
    for key in sorted(gbest.keys()):
        print(f"{key:<20} {gbest[key]:>8.3f} +- {gerr[key]:.3f}")

if __name__ == "__main__":
    main()
