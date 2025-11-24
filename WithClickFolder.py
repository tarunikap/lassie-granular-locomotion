import pandas as pd
import matplotlib.pyplot as plt
import os

clicked_times = []

def interactive_click_plot(df, time_col, fx_col, fy_col, toe_y_col, file_path, file_index, total_files):
    global clicked_times
    clicked_times.clear()

    filename = os.path.basename(file_path)
    print(f"\nProcessing file {file_index + 1} of {total_files}: {filename}")
    print("Click anywhere on the plot 4 times (Penetration Start to End, Extraction Start to End)")

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()

    ax1.plot(df[time_col], df[fx_col], label='Force Fx', color='blue')
    ax1.plot(df[time_col], df[fy_col], label='Force Fy', color='red')
    ax2.plot(df[time_col], df[toe_y_col], label='Toe Y Position', color='orange')

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Force (N)")
    ax2.set_ylabel("Toe Y Position (m)")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f"{filename}\nClick: 1. Penetration Start  2. End  3. Extraction Start  4. End")

    def on_click(event):
        x = event.xdata
        if x is None:
            return
        idx = (df[time_col] - x).abs().idxmin()
        time_val = df[time_col].iloc[idx]
        clicked_times.append(time_val)

        print(f"Click {len(clicked_times)}: {time_val:.3f} s")

        ax1.axvline(time_val, linestyle='--', color='green' if len(clicked_times) <= 2 else 'magenta')
        ax1.text(time_val, ax1.get_ylim()[1]*0.9, str(len(clicked_times)), fontsize=10,
                 color='black', ha='center', va='top', backgroundcolor='white')
        fig.canvas.draw_idle()

        if len(clicked_times) == 4:
            plt.close(fig)
            review_and_confirm(df, time_col, fx_col, fy_col, toe_y_col, clicked_times.copy(), file_path)

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

def review_and_confirm(df, time_col, fx_col, fy_col, toe_y_col, cut_times, file_path):
    t1, t2, t3, t4 = cut_times
    filename = os.path.basename(file_path)

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()

    ax1.plot(df[time_col], df[fx_col], label='Force Fx', color='blue')
    ax1.plot(df[time_col], df[fy_col], label='Force Fy', color='red')
    ax2.plot(df[time_col], df[toe_y_col], label='Toe Y Position', color='orange')

    labels = ['1', '2', '3', '4']
    colors = ['green', 'green', 'magenta', 'magenta']
    for t, label, color in zip(cut_times, labels, colors):
        ax1.axvline(t, linestyle='--', color=color)
        ax1.text(t, ax1.get_ylim()[1]*0.9, label, rotation=0, fontsize=10,
                 color=color, ha='center', va='top', backgroundcolor='white')

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Force (N)")
    ax2.set_ylabel("Toe Y Position (m)")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f"Review Cut Points: {filename}")

    # Save review plot
    reviewplots_dir = r"C:\Users\Tarunika P\Desktop\Summer '25 Intern\USC\LASSIE\LASSIE_DataForHeatMap\LASSIE_DataForHeatMap\reviewplots"
    os.makedirs(reviewplots_dir, exist_ok=True)
    review_img_path = os.path.join(reviewplots_dir, f"review_{os.path.splitext(filename)[0]}.png")
    plt.savefig(review_img_path, dpi=300)
    print(f"\nReview plot saved: {review_img_path}")

    plt.show()

    cont = input("\nSave data? (yes / no ): ").strip().lower()
    if cont == 'yes':
        save_cut_csvs(df, time_col, cut_times, file_path)
    else:
        interactive_click_plot(df, time_col, fx_col, fy_col, toe_y_col, file_path, -1, -1)  # retry

def save_cut_csvs(df, time_col, cut_times, file_path):
    t1, t2, t3, t4 = cut_times
    base = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = r"C:\Users\Tarunika P\Desktop\Summer '25 Intern\USC\LASSIE\LASSIE_DataForHeatMap\LASSIE_DataForHeatMap\cut files\cut_plate30"
    os.makedirs(out_dir, exist_ok=True)

    df_pen = df[(df[time_col] >= t1) & (df[time_col] <= t2)]
    df_ext = df[(df[time_col] >= t3) & (df[time_col] <= t4)]

    pen_path = os.path.join(out_dir, f"cut_penetration_{base}.csv")
    ext_path = os.path.join(out_dir, f"cut_extraction_{base}.csv")

    df_pen.to_csv(pen_path, index=False)
    df_ext.to_csv(ext_path, index=False)

    print(f"Penetration saved: {pen_path}")
    print(f"Extraction saved: {ext_path}")

def main():
    #folder path
    folder_path = r"C:\Users\Tarunika P\Desktop\Summer '25 Intern\USC\LASSIE\LASSIE_DataForHeatMap\LASSIE_DataForHeatMap\plate30\plate30ex_angleNeg60"

    files = [os.path.join(folder_path, f)
             for f in os.listdir(folder_path)
             if f.endswith(".csv") and not f.startswith("cut")]

    total_files = len(files)
    if total_files == 0:
        print("No valid .csv files found in folder.")
        return

    for idx, file_path in enumerate(files):
        print(f"\n--- File {idx + 1} of {total_files} ---")
        df = pd.read_csv(file_path, skiprows=2)

        time_col = 'time'
        fx_col = 'toeforce_x'
        fy_col = 'toeforce_y'
        toe_y_col = 'toe_position_y'

        interactive_click_plot(df, time_col, fx_col, fy_col, toe_y_col, file_path, idx, total_files)

if __name__ == "__main__":
    main()
