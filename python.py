import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# Constants (atomic units: ℏ=1, m=1, L=1)
L = 1.0
N_MAX = 15  # Default number of basis states

def unperturbed_wavefunction(n, x):
    """Unperturbed wavefunction for infinite square well: ψ_n(x) = √(2/L)sin(nπx/L)"""
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

def unperturbed_energy(n):
    """Unperturbed energy: E_n = n²π²ℏ²/(2mL²) = n²π²/2 in atomic units"""
    return n**2 * np.pi**2 / 2.0

def perturbation_potential(x, lambda_val):
    """Perturbation: H'(x) = λ for 0 ≤ x ≤ L/2, else 0"""
    return lambda_val * (x <= L/2).astype(float) if hasattr(x, '__len__') else (lambda_val if x <= L/2 else 0.0)

def calculate_matrix_element(k, n, lambda_val):
    """
    Matrix element H'_kn = ⟨ψ_k|H'|ψ_n⟩
    
    Analytical result for half-well constant perturbation:
    - Diagonal (k=n): λ/2
    - Off-diagonal (k≠n): (λ/π)[sin((k-n)π/2)/(k-n) - sin((k+n)π/2)/(k+n)]
    """
    k, n = int(k), int(n)
    
    if k == n:
        return lambda_val / 2.0
    else:
        term1 = np.sin((k - n) * np.pi / 2) / (k - n)
        term2 = np.sin((k + n) * np.pi / 2) / (k + n)
        return (lambda_val / np.pi) * (term1 - term2)

def normalize_wavefunction(psi, x):
    """Normalize wavefunction to ensure ∫|ψ|²dx = 1"""
    dx = x[1] - x[0]
    norm_squared = np.trapz(psi**2, dx=dx)
    return psi / np.sqrt(norm_squared) if norm_squared > 0 else psi

def calculate_first_order_correction(n, x_array, lambda_val, n_max=N_MAX):
    """
    First-order wavefunction correction:
    ψ_n^(1) = Σ_{k≠n} [H'_kn / (E_n^(0) - E_k^(0))] ψ_k^(0)
    """
    psi_n_1 = np.zeros_like(x_array, dtype=float)
    E_n_0 = unperturbed_energy(n)
    
    for k in range(1, n_max + 1):
        if k == n:
            continue
        
        E_k_0 = unperturbed_energy(k)
        H_kn = calculate_matrix_element(k, n, lambda_val)
        
        c_k = H_kn / (E_n_0 - E_k_0)
        psi_k_0 = unperturbed_wavefunction(k, x_array)
        psi_n_1 += c_k * psi_k_0
    
    return psi_n_1

def calculate_second_order_correction(n, x_array, lambda_val, n_max=N_MAX):
    """
    Second-order wavefunction correction (simplified):
    ψ_n^(2) contains terms involving products of matrix elements
    """
    psi_n_2 = np.zeros_like(x_array, dtype=float)
    E_n_0 = unperturbed_energy(n)
    H_nn = calculate_matrix_element(n, n, lambda_val)
    
    for k in range(1, min(n_max, 10) + 1):  # Limit for computational efficiency
        if k == n:
            continue
        
        E_k_0 = unperturbed_energy(k)
        H_kn = calculate_matrix_element(k, n, lambda_val)
        
        # Second-order correction term
        inner_sum = 0.0
        for m in range(1, min(n_max, 10) + 1):
            if m == n:
                continue
            E_m_0 = unperturbed_energy(m)
            H_km = calculate_matrix_element(k, m, lambda_val)
            H_mn = calculate_matrix_element(m, n, lambda_val)
            inner_sum += H_km * H_mn / (E_n_0 - E_m_0)
        
        c_k = (inner_sum / (E_n_0 - E_k_0)) - (H_nn * H_kn / (E_n_0 - E_k_0)**2)
        psi_k_0 = unperturbed_wavefunction(k, x_array)
        psi_n_2 += c_k * psi_k_0
    
    return psi_n_2

def calculate_energy_corrections(n, lambda_val, n_max=N_MAX):
    """
    Calculate energy corrections:
    E_n^(1) = ⟨ψ_n^(0)|H'|ψ_n^(0)⟩ = H'_nn
    E_n^(2) = Σ_{k≠n} |H'_kn|² / (E_n^(0) - E_k^(0))
    """
    # First-order
    E_1 = calculate_matrix_element(n, n, lambda_val)
    
    # Second-order
    E_2 = 0.0
    E_n_0 = unperturbed_energy(n)
    
    for k in range(1, n_max + 1):
        if k == n:
            continue
        H_kn = calculate_matrix_element(k, n, lambda_val)
        E_k_0 = unperturbed_energy(k)
        E_2 += H_kn**2 / (E_n_0 - E_k_0)
    
    return E_1, E_2

def calculate_position_statistics(psi, x):
    """Calculate ⟨x⟩ and Δx = √(⟨x²⟩ - ⟨x⟩²)"""
    dx = x[1] - x[0]
    prob = psi**2
    
    x_avg = np.trapz(x * prob, dx=dx)
    x2_avg = np.trapz(x**2 * prob, dx=dx)
    delta_x = np.sqrt(x2_avg - x_avg**2)
    
    return x_avg, delta_x

def check_perturbation_validity(n, lambda_val, n_max=N_MAX):
    """
    Check validity: requires |H'_kn| << |E_n - E_k|
    Returns maximum ratio across all k≠n
    """
    E_n = unperturbed_energy(n)
    max_ratio = 0.0
    worst_k = None
    
    for k in range(1, n_max + 1):
        if k == n:
            continue
        E_k = unperturbed_energy(k)
        H_kn = calculate_matrix_element(k, n, lambda_val)
        ratio = abs(H_kn / (E_n - E_k))
        
        if ratio > max_ratio:
            max_ratio = ratio
            worst_k = k
    
    return max_ratio, worst_k

def solve_exact_hamiltonian(lambda_val, n_states=10):
    """
    Solve the full Hamiltonian exactly via matrix diagonalization.
    H_mn = E_n δ_mn + H'_mn
    """
    H_matrix = np.zeros((n_states, n_states))
    
    for i in range(n_states):
        for j in range(n_states):
            n_i, n_j = i + 1, j + 1
            if i == j:
                H_matrix[i, j] = unperturbed_energy(n_i) + calculate_matrix_element(n_i, n_j, lambda_val)
            else:
                H_matrix[i, j] = calculate_matrix_element(n_i, n_j, lambda_val)
    
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    return eigenvalues, eigenvectors

def comprehensive_analysis(lambda_val, state_n, n_max=N_MAX, include_second_order=False):
    """Perform comprehensive perturbation analysis"""
    
    print(f"\n{'='*70}")
    print(f"QUANTUM PERTURBATION THEORY ANALYSIS")
    print(f"{'='*70}")
    print(f"State: n = {state_n}")
    print(f"Perturbation strength: λ = {lambda_val}")
    print(f"Basis states included: N_max = {n_max}")
    print(f"{'='*70}\n")
    
    # Position array
    x = np.linspace(0, L, 1000)
    
    # 1. UNPERTURBED STATE
    psi_0 = unperturbed_wavefunction(state_n, x)
    E_0 = unperturbed_energy(state_n)
    x_avg_0, delta_x_0 = calculate_position_statistics(psi_0, x)
    
    print("1. UNPERTURBED STATE")
    print(f"   Energy E_{state_n}^(0) = {E_0:.6f}")
    print(f"   ⟨x⟩ = {x_avg_0:.6f}")
    print(f"   Δx = {delta_x_0:.6f}")
    
    # 2. PERTURBATION VALIDITY CHECK
    print(f"\n2. PERTURBATION THEORY VALIDITY")
    max_ratio, worst_k = check_perturbation_validity(state_n, lambda_val, n_max)
    print(f"   Max |H'_kn/(E_n - E_k)| = {max_ratio:.6f} (k = {worst_k})")
    
    if max_ratio < 0.1:
        print(f"   ✓ EXCELLENT: Perturbation theory is highly reliable")
    elif max_ratio < 0.3:
        print(f"   ✓ GOOD: Perturbation theory should be accurate")
    elif max_ratio < 0.5:
        print(f"   ⚠ MARGINAL: Results may have moderate errors")
    else:
        print(f"   ✗ POOR: Perturbation theory may not be reliable!")
    
    # 3. ENERGY CORRECTIONS
    E_1, E_2 = calculate_energy_corrections(state_n, lambda_val, n_max)
    print(f"\n3. ENERGY CORRECTIONS")
    print(f"   First-order:  E_{state_n}^(1) = {E_1:.6f}")
    print(f"   Second-order: E_{state_n}^(2) = {E_2:.6f}")
    print(f"   Total (1st):  E_{state_n} ≈ {E_0 + E_1:.6f}")
    print(f"   Total (2nd):  E_{state_n} ≈ {E_0 + E_1 + E_2:.6f}")
    
    # 4. WAVEFUNCTION CORRECTIONS
    print(f"\n4. WAVEFUNCTION CORRECTIONS")
    psi_1 = calculate_first_order_correction(state_n, x, lambda_val, n_max)
    psi_pert_1st = psi_0 + psi_1
    psi_pert_1st = normalize_wavefunction(psi_pert_1st, x)
    
    if include_second_order:
        psi_2 = calculate_second_order_correction(state_n, x, lambda_val, n_max)
        psi_pert_2nd = psi_0 + psi_1 + psi_2
        psi_pert_2nd = normalize_wavefunction(psi_pert_2nd, x)
        
        correction_norm_2 = np.sqrt(np.trapz(psi_2**2, dx=(x[1]-x[0])))
        print(f"   ||ψ^(2)|| = {correction_norm_2:.6f}")
    else:
        psi_pert_2nd = None
    
    correction_norm_1 = np.sqrt(np.trapz(psi_1**2, dx=(x[1]-x[0])))
    print(f"   ||ψ^(1)|| = {correction_norm_1:.6f}")
    
    # 5. EXPECTATION VALUES (PERTURBED)
    x_avg_1, delta_x_1 = calculate_position_statistics(psi_pert_1st, x)
    print(f"\n5. PERTURBED STATE (1st order)")
    print(f"   ⟨x⟩ = {x_avg_1:.6f} (shift: {x_avg_1 - x_avg_0:+.6f})")
    print(f"   Δx = {delta_x_1:.6f} (change: {delta_x_1 - delta_x_0:+.6f})")
    
    # 6. EXACT SOLUTION COMPARISON
    print(f"\n6. EXACT SOLUTION (via diagonalization)")
    eigenvalues, eigenvectors = solve_exact_hamiltonian(lambda_val, min(n_max, 10))
    E_exact = eigenvalues[state_n - 1]
    print(f"   Exact E_{state_n} = {E_exact:.6f}")
    print(f"   Error (1st order): {abs(E_0 + E_1 - E_exact):.6e}")
    print(f"   Error (2nd order): {abs(E_0 + E_1 + E_2 - E_exact):.6e}")
    
    # Reconstruct exact wavefunction
    psi_exact = np.zeros_like(x)
    for k in range(len(eigenvectors)):
        psi_exact += eigenvectors[k, state_n - 1] * unperturbed_wavefunction(k + 1, x)
    psi_exact = normalize_wavefunction(psi_exact, x)
    
    # Calculate overlap/fidelity
    dx = x[1] - x[0]
    fidelity = abs(np.trapz(np.conj(psi_pert_1st) * psi_exact, dx=dx))**2
    print(f"   Fidelity |⟨ψ_pert|ψ_exact⟩|² = {fidelity:.6f}")
    
    print(f"\n{'='*70}\n")
    
    return {
        'x': x,
        'psi_0': psi_0,
        'psi_1': psi_1,
        'psi_pert_1st': psi_pert_1st,
        'psi_pert_2nd': psi_pert_2nd,
        'psi_exact': psi_exact,
        'E_0': E_0,
        'E_1': E_1,
        'E_2': E_2,
        'E_exact': E_exact,
        'eigenvalues': eigenvalues
    }

def create_comprehensive_plot(results, lambda_val, state_n):
    """Create comprehensive multi-panel visualization"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    x = results['x']
    
    # Panel 1: Wavefunctions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, results['psi_0'], 'k--', linewidth=2, label=r'$\psi_n^{(0)}$ (Unperturbed)', alpha=0.7)
    ax1.plot(x, results['psi_pert_1st'], 'b-', linewidth=2, label=r'$\psi_n$ (1st order)')
    ax1.plot(x, results['psi_exact'], 'r:', linewidth=2, label=r'$\psi_n$ (Exact)')
    ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.axvline(L/2, color='red', linestyle=':', alpha=0.3, label='Perturbation edge')
    ax1.set_xlabel('Position x/L', fontsize=11)
    ax1.set_ylabel(r'Wavefunction $\psi(x)$', fontsize=11)
    ax1.set_title(f'(a) Wavefunctions (n={state_n}, λ={lambda_val})', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Probability Densities
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, results['psi_0']**2, 'k--', linewidth=2, label=r'$|\psi_n^{(0)}|^2$', alpha=0.7)
    ax2.plot(x, results['psi_pert_1st']**2, 'b-', linewidth=2, label=r'$|\psi_n|^2$ (1st order)')
    ax2.plot(x, results['psi_exact']**2, 'r:', linewidth=2, label=r'$|\psi_n|^2$ (Exact)')
    
    # Add perturbation potential (scaled)
    V = perturbation_potential(x, lambda_val)
    V_scaled = V * np.max(results['psi_0']**2) / (2 * lambda_val) if lambda_val != 0 else V
    ax2.fill_between(x, 0, V_scaled, alpha=0.2, color='red', label="H'(x) (scaled)")
    
    ax2.set_xlabel('Position x/L', fontsize=11)
    ax2.set_ylabel(r'Probability Density $|\psi|^2$', fontsize=11)
    ax2.set_title(f'(b) Probability Densities', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Panel 3: First-order Correction
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x, results['psi_1'], 'g-', linewidth=2, label=r'$\psi_n^{(1)}$ (correction)')
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax3.axvline(L/2, color='red', linestyle=':', alpha=0.3)
    ax3.fill_between(x, 0, results['psi_1'], alpha=0.3, color='green')
    ax3.set_xlabel('Position x/L', fontsize=11)
    ax3.set_ylabel(r'$\psi_n^{(1)}(x)$', fontsize=11)
    ax3.set_title('(c) First-Order Wavefunction Correction', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Difference (Exact - Perturbation)
    ax4 = fig.add_subplot(gs[1, 1])
    diff = results['psi_exact'] - results['psi_pert_1st']
    ax4.plot(x, diff, 'purple', linewidth=2, label='Error: Exact - 1st order')
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax4.axvline(L/2, color='red', linestyle=':', alpha=0.3)
    ax4.fill_between(x, 0, diff, alpha=0.3, color='purple')
    ax4.set_xlabel('Position x/L', fontsize=11)
    ax4.set_ylabel(r'$\Delta\psi(x)$', fontsize=11)
    ax4.set_title('(d) Perturbation Theory Error', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Energy Level Diagram
    ax5 = fig.add_subplot(gs[2, 0])
    n_levels = min(5, len(results['eigenvalues']))
    
    for n in range(1, n_levels + 1):
        E0 = unperturbed_energy(n)
        E1, E2 = calculate_energy_corrections(n, lambda_val, N_MAX)
        E_pert = E0 + E1 + E2
        E_exact = results['eigenvalues'][n-1]
        
        # Unperturbed
        ax5.plot([0, 0.8], [E0, E0], 'k-', linewidth=2, alpha=0.5)
        # Perturbed (1st+2nd order)
        ax5.plot([1.2, 2.0], [E_pert, E_pert], 'b-', linewidth=2)
        # Exact
        ax5.plot([2.2, 3.0], [E_exact, E_exact], 'r-', linewidth=2)
        
        # Highlight the state being analyzed
        if n == state_n:
            ax5.plot([0, 0.8], [E0, E0], 'k-', linewidth=4)
            ax5.plot([1.2, 2.0], [E_pert, E_pert], 'b-', linewidth=4)
            ax5.plot([2.2, 3.0], [E_exact, E_exact], 'r-', linewidth=4)
            
            # Connect with arrows
            ax5.annotate('', xy=(1.2, E_pert), xytext=(0.8, E0),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, alpha=0.5))
        
        ax5.text(-0.3, E0, f'n={n}', fontsize=9, va='center')
    
    ax5.set_xlim(-0.5, 3.5)
    ax5.set_ylabel('Energy', fontsize=11)
    ax5.set_title('(e) Energy Level Diagram', fontsize=12, fontweight='bold')
    ax5.set_xticks([0.4, 1.6, 2.6])
    ax5.set_xticklabels(['Unperturbed', 'Perturbation\n(1st+2nd)', 'Exact'])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Panel 6: Convergence Analysis
    ax6 = fig.add_subplot(gs[2, 1])
    
    n_terms_list = range(1, 21)
    energies_pert = []
    errors = []
    
    for nt in n_terms_list:
        E1, E2 = calculate_energy_corrections(state_n, lambda_val, nt)
        E_pert = results['E_0'] + E1 + E2
        energies_pert.append(E_pert)
        errors.append(abs(E_pert - results['E_exact']))
    
    ax6.semilogy(n_terms_list, errors, 'o-', linewidth=2, markersize=6, color='darkgreen')
    ax6.axhline(1e-6, color='red', linestyle='--', alpha=0.5, label='10⁻⁶ threshold')
    ax6.set_xlabel('Number of basis states (N)', fontsize=11)
    ax6.set_ylabel('Energy Error |E_pert - E_exact|', fontsize=11)
    ax6.set_title('(f) Convergence of Perturbation Theory', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, which='both')
    ax6.legend(fontsize=9)
    
    plt.suptitle(f'Comprehensive Quantum Perturbation Analysis: n={state_n}, λ={lambda_val}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    return fig

def get_interactive_input():
    """Interactive mode to get parameters from user"""
    print("\n" + "="*70)
    print("QUANTUM PERTURBATION THEORY - INTERACTIVE MODE")
    print("="*70)
    
    # Get lambda value
    print("\n📊 Perturbation Strength (λ):")
    print("   Recommended values:")
    print("   • 0.1  - Small perturbation (best accuracy)")
    print("   • 0.5  - Medium perturbation (clear effects)")
    print("   • 1.0  - Large perturbation (pushes limits)")
    print("   • -0.3 - Negative perturbation")
    
    while True:
        try:
            lambda_input = input("\nEnter λ value (or press Enter for default 0.1): ").strip()
            lambda_val = 0.1 if lambda_input == "" else float(lambda_input)
            break
        except ValueError:
            print("❌ Invalid input! Please enter a number.")
    
    # Get state
    print("\n🔬 Quantum State:")
    print("   • 1 - Ground state (lowest energy)")
    print("   • 2 - First excited state")
    print("   • 3 - Second excited state")
    print("   • 4 - Third excited state")
    
    while True:
        try:
            state_input = input("\nEnter state n (or press Enter for default 1): ").strip()
            state_n = 1 if state_input == "" else int(state_input)
            if state_n in [1, 2, 3, 4]:
                break
            else:
                print("❌ Please choose 1, 2, 3, or 4")
        except ValueError:
            print("❌ Invalid input! Please enter an integer.")
    
    # Get nmax
    print("\n⚙️  Advanced Options:")
    nmax_input = input("Number of basis states (or press Enter for default 15): ").strip()
    nmax = 15 if nmax_input == "" else int(nmax_input)
    
    # Get second-order option
    second_input = input("Include second-order corrections? (y/n, default: n): ").strip().lower()
    include_second = second_input == 'y'
    
    print("\n" + "="*70 + "\n")
    
    return lambda_val, state_n, nmax, include_second

def main():
    # Check if running in interactive mode (no arguments provided)
    if len(sys.argv) == 1:
        # Interactive mode
        lambda_val, state_n, nmax, include_second = get_interactive_input()
        output_path = None
    else:
        # Command-line argument mode
        parser = argparse.ArgumentParser(
            description="Comprehensive quantum mechanical perturbation theory analysis",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python %(prog)s 0.1 --state 1
  python %(prog)s 0.5 --state 2 --nmax 20 --second-order
  python %(prog)s -0.3 --state 1 --nmax 15

Interactive Mode (no arguments):
  python %(prog)s
            """
        )
        
        parser.add_argument('lambda_val', type=float, nargs='?', default=None,
                           help="Perturbation strength λ (try 0.1 to 1.0 for good results)")
        parser.add_argument('--state', type=int, default=1, choices=[1, 2, 3, 4],
                           help="Quantum state n (default: 1)")
        parser.add_argument('--nmax', type=int, default=15,
                           help="Number of basis states (default: 15)")
        parser.add_argument('--second-order', action='store_true',
                           help="Include second-order corrections (slower)")
        parser.add_argument('--output', type=str, default=None,
                           help="Output filename (default: auto-generated)")
        
        args = parser.parse_args()
        
        if args.lambda_val is None:
            # If lambda_val not provided, switch to interactive mode
            lambda_val, state_n, nmax, include_second = get_interactive_input()
            output_path = None
        else:
            lambda_val = args.lambda_val
            state_n = args.state
            nmax = args.nmax
            include_second = args.second_order
            output_path = args.output
    
    # Validate lambda value
    if abs(lambda_val) > 2.0:
        print(f"\n⚠️  WARNING: |λ|={abs(lambda_val)} is large!")
        print("   Perturbation theory works best for |λ| < 1.0")
        response = input("   Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting...")
            sys.exit(0)
    
    try:
        # Perform analysis
        results = comprehensive_analysis(
            lambda_val, 
            state_n, 
            nmax,
            include_second
        )
        
        # Create visualization
        print("📊 Generating plots...")
        fig = create_comprehensive_plot(results, lambda_val, state_n)
        
        # Save figure
        if output_path is None:
            safe_lambda = str(lambda_val).replace('.', 'p').replace('-', 'm')
            output_path = f'perturbation_analysis_n{state_n}_lambda_{safe_lambda}.png'
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ SUCCESS!")
        print(f"📁 Plot saved: {output_path}")
        print(f"🔬 Analysis complete!\n")
        
        plt.close()
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()