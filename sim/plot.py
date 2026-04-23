import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def plot_joint_angles(csv_filename):
    """Read joint data from CSV and plot angles vs time, save as PDF"""
    # Read data from CSV
    times = []
    joint1_angles = []
    joint2_angles = []
    joint3_angles = []
    joint4_angles = []
    
    with open(csv_filename, mode='r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            times.append(float(row[0]))
            joint1_angles.append(float(row[1]))
            joint2_angles.append(float(row[2]))
            joint3_angles.append(float(row[3]))
            joint4_angles.append(float(row[4]))
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(times, joint1_angles, label='Joint 1 (Left Hip)', linewidth=2)
    plt.plot(times, joint2_angles, label='Joint 2 (Left Ankle)', linewidth=2)
    plt.plot(times, joint3_angles, label='Joint 3 (Right Hip)', linewidth=2)
    plt.plot(times, joint4_angles, label='Joint 4 (Right Ankle)', linewidth=2)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Angle (radians)', fontsize=12)
    plt.title('Joint Angles vs Time', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save as PDF
    pdf_filename = csv_filename.replace('.csv', '_plot.pdf')
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig()
        plt.close()
    
    print(f"Plot saved to {os.path.abspath(pdf_filename)}")

if __name__ == "__main__":
    csv_filename = "./joint_data_log.csv"
    plot_joint_angles(csv_filename)
