import pandas as pd
import matplotlib.pyplot as plt
import os

def interactive_plot_with_cursor_info(df, time_col, fx_col, fy_col, toe_y_col, cut_times=None):
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()

    ax1.plot(df[time_col], df[fx_col], label='Force Fx', color='blue')
    ax1.plot(df[time_col], df[fy_col], label='Force Fy', color='red')
    ax2.plot(df[time_col], df[toe_y_col], label='Toe Y Position', color='orange')

    if cut_times:
        labels = ['Penetration Start', 'Penetration End', 'Extraction Start', 'Extraction End']
        colors = ['green', 'black', 'magenta', 'gray']
        for t, label, color in zip(cut_times, labels, colors):
            ax1.axvline(t, color=color, linestyle='--', label=label)

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Force (N)")
    ax2.set_ylabel("Toe Y Position (m)")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title("Interactive Plot: Move mouse to see time and value")

    text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def on_mouse_move(event):
        if event.inaxes == ax1 or event.inaxes == ax2:
            xdata = event.xdata
            if xdata is None:
                return
            idx = (df[time_col] - xdata).abs().idxmin()
            time_val = df[time_col].iloc[idx]
            fx_val = df[fx_col].iloc[idx]
            fy_val = df[fy_col].iloc[idx]
            toe_val = df[toe_y_col].iloc[idx]
            text.set_text(f"Time: {time_val:.3f} s\nFx: {fx_val:.3f} N\nFy: {fy_val:.3f} N\nToe Y: {toe_val:.3f} m")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    plt.show(block=False)

def main():
    file_path = r"C:\Users\Tarunika P\Desktop\Summer '25 Intern\USC\LASSIE\LASSIE_DataForHeatMap\LASSIE_DataForHeatMap\plate1ex_angle0\plate1ex_angle0ex_len10_t1_Fri_Jun_13_15_16_07_2025.csv"
    df = pd.read_csv(file_path, skiprows=2)

    time_col = 'time'
    fx_col = 'toeforce_x'
    fy_col = 'toeforce_y'
    toe_y_col = 'toe_position_y'

    interactive_plot_with_cursor_info(df, time_col, fx_col, fy_col, toe_y_col)

    print("\nMove your cursor over the plot to view time and values.\n Now enter the time values below:")

    t1 = float(input("Penetration Start: "))
    t2 = float(input("Penetration End: "))
    t3 = float(input("Extraction Start: "))
    t4 = float(input("Extraction End: "))
    cut_times = [t1, t2, t3, t4]

    interactive_plot_with_cursor_info(df, time_col, fx_col, fy_col, toe_y_col, cut_times)

    cont = input("\nPlot updated. Save data and exit? (yes / no to re-enter times): ").strip().lower()
    if cont == 'yes':
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = r"C:\Users\Tarunika P\Desktop\Summer '25 Intern\USC\LASSIE\LASSIE_DataForHeatMap\LASSIE_DataForHeatMap\cut files"
        os.makedirs(output_dir, exist_ok=True)

        df_penetration = df[(df[time_col] >= t1) & (df[time_col] <= t2)]
        penetration_path = os.path.join(output_dir, f"cut_penetration_{base_name}.csv")
        df_penetration.to_csv(penetration_path, index=False)

        df_extraction = df[(df[time_col] >= t3) & (df[time_col] <= t4)]
        extraction_path = os.path.join(output_dir, f"cut_extraction_{base_name}.csv")
        df_extraction.to_csv(extraction_path, index=False)

        print(f"\nPenetration data saved to: {penetration_path}")
        print(f"Extraction data saved to: {extraction_path}")
    else:
        print("\nRerun the script to enter new values.")

if __name__ == "__main__":
    main()
