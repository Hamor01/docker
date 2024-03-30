import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import os

class MockedModel:
    @staticmethod
    def predict(data_point):
        return np.array([[random.uniform(5, 20), random.uniform(10, 20)]])

class MockedRNNModel:
    @staticmethod
    def predict(data_point):
        return np.array([[random.uniform(5, 20), random.uniform(10, 20)]])

class MockedGRUModel:
    @staticmethod
    def predict(data_point):
        return np.array([[random.uniform(5, 20), random.uniform(10, 20)]])

class DataCollectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Collection and AI Integration")

        # Initialize data storage
        self.data = pd.DataFrame(columns=['Patient_ID', 'Electrode', 'Distance', 'Amperage', 'AI_Distance', 'AI_Amperage'])



        # GUI components
        self.patient_id_label = ttk.Label(root, text="Patient ID:")
        self.patient_id_entry = ttk.Entry(root)
        self.electrode_label = ttk.Label(root, text="Select Electrode:")
        self.electrode_combobox = ttk.Combobox(root, values=["Electro A", "Electro B"])
        self.distance_label = ttk.Label(root, text="Select Distance:")
        self.distance_slider = ttk.Scale(root, from_=5, to=20, orient="horizontal", length=200, command=self.update_distance_label)
        self.distance_value_label = ttk.Label(root, text="Current Distance: 5")  
        self.amperage_label = ttk.Label(root, text="Select Amperage:")
        self.amperage_slider = ttk.Scale(root, from_=10, to=20, orient="horizontal", length=200, command=self.update_amperage_label)
        self.amperage_value_label = ttk.Label(root, text="Current Amperage: 10")  

        self.record_button = ttk.Button(root, text="Record Data", command=self.record_data)
        self.predict_button = ttk.Button(root, text="Get AI Prediction", command=self.get_ai_prediction)
        self.result_label = ttk.Label(root, text="AI Prediction: ")
        
        self.simulate_button = ttk.Button(root, text="Simulate Treatment", command=self.simulate_treatment)
        self.save_button = ttk.Button(root, text="Save Data and Output", command=self.save_data_and_output)

        self.virtual_planning_label = ttk.Label(root, text="Virtual Planning:")
        self.virtual_planning_text = tk.Text(root, height=5, width=50)
        
        self.waveform_frequency_label = ttk.Label(root, text="Waveform Frequency:")
        self.waveform_frequency_entry = ttk.Entry(root)
        self.waveform_amplitude_label = ttk.Label(root, text="Waveform Amplitude:")
        self.waveform_amplitude_entry = ttk.Entry(root)
        self.waveform_pulse_width_label = ttk.Label(root, text="Waveform Pulse Width:")
        self.waveform_pulse_width_entry = ttk.Entry(root)
        
        self.virtual_planning_button = ttk.Button(root, text="Propose Virtual Planning", command=self.propose_virtual_planning)

        # GUI layout
        self.patient_id_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.patient_id_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.electrode_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.electrode_combobox.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        self.distance_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.distance_slider.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        self.distance_value_label.grid(row=2, column=2, padx=10, pady=10, sticky="w")  
        self.amperage_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.amperage_slider.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        self.amperage_value_label.grid(row=3, column=2, padx=10, pady=10, sticky="w")

        self.record_button.grid(row=4, column=0, columnspan=2, pady=10)
        self.predict_button.grid(row=5, column=0, columnspan=2, pady=10)
        self.result_label.grid(row=6, column=0, columnspan=2, pady=10)
        
        self.virtual_planning_label.grid(row=7, column=0, columnspan=2, pady=10, sticky="w")
        self.virtual_planning_text.grid(row=8, column=0, columnspan=2, pady=10, sticky="w")

        self.waveform_frequency_label.grid(row=9, column=0, padx=10, pady=10, sticky="w")
        self.waveform_frequency_entry.grid(row=9, column=1, padx=10, pady=10, sticky="w")
        self.waveform_amplitude_label.grid(row=10, column=0, padx=10, pady=10, sticky="w")
        self.waveform_amplitude_entry.grid(row=10, column=1, padx=10, pady=10, sticky="w")
        self.waveform_pulse_width_label.grid(row=11, column=0, padx=10, pady=10, sticky="w")
        self.waveform_pulse_width_entry.grid(row=11, column=1, padx=10, pady=10, sticky="w")
        
        self.virtual_planning_button.grid(row=12, column=0, columnspan=2, pady=10)

        # Graphs
        self.scatter_plot_button = ttk.Button(root, text="Show Scatter Plot", command=self.show_scatter_plot)
        self.distance_histogram_button = ttk.Button(root, text="Show Distance Histogram", command=self.show_distance_histogram)
        self.amperage_histogram_button = ttk.Button(root, text="Show Amperage Histogram", command=self.show_amperage_histogram)
        self.compare_distance_chart_button = ttk.Button(root, text="Compare AI vs Actual Distance", command=self.compare_distance_chart)
        self.compare_amperage_chart_button = ttk.Button(root, text="Compare AI vs Actual Amperage", command=self.compare_amperage_chart)

        self.scatter_plot_button.grid(row=13, column=0, columnspan=2, pady=10)
        self.distance_histogram_button.grid(row=14, column=0, columnspan=2, pady=10)
        self.amperage_histogram_button.grid(row=15, column=0, columnspan=2, pady=10)
        self.compare_distance_chart_button.grid(row=16, column=0, columnspan=2, pady=10)
        self.compare_amperage_chart_button.grid(row=17, column=0, columnspan=2, pady=10)
        
        # Mocked AI model instances
        self.model = MockedModel()
        self.rnn_model = MockedRNNModel()
        self.gru_model = MockedGRUModel()

        # Configure column and row weights for flexibility
        # Configure column and row weights for flexibility
        for i in range(17):  # rows
            self.root.grid_rowconfigure(i, weight=1)
        for i in range(3):  # columns
            self.root.grid_columnconfigure(i, weight=1)

        # Additional weights for specific rows
        self.root.grid_rowconfigure(8, weight=1)  # Increase weight for row with virtual_planning_text
        self.root.grid_rowconfigure(12, weight=1)  # Increase weight for row with virtual_planning_button

        # Additional adjustments for button and entry sizes, font sizes, and padding
        ttk.Style().configure('TButton', font=('Arial', 12))  # Adjust button font size
        ttk.Style().configure('TEntry', font=('Arial', 12))  # Adjust entry font size
        self.virtual_planning_text.config(font=('Arial', 12))  # Adjust text widget font size

        # Adjust button size and padding
        self.record_button.config(width=20, padding=(5, 5))
        self.predict_button.config(width=20, padding=(5, 5))
        self.simulate_button.config(width=20, padding=(5, 5))
        self.save_button.config(width=20, padding=(5, 5))
        self.virtual_planning_button.config(width=20, padding=(5, 5))

        # Adjust entry size
        self.patient_id_entry.config(width=20)
        self.waveform_frequency_entry.config(width=20)
        self.waveform_amplitude_entry.config(width=20)
        self.waveform_pulse_width_entry.config(width=20)

    def update_distance_label(self, value):
        self.distance_value_label.config(text=f"Current Distance: {value}")

    def update_amperage_label(self, value):
        self.amperage_value_label.config(text=f"Current Amperage: {value}")

    def record_data(self):
        patient_id = self.patient_id_entry.get()
        electrode_type = self.electrode_combobox.get()
        distance = self.distance_slider.get()
        amperage = self.amperage_slider.get()

        # Save the collected data
        data_row = pd.Series({'Patient_ID': patient_id, 'Electrode': electrode_type, 'Distance': distance, 'Amperage': amperage})
        self.data = pd.concat([self.data, data_row.to_frame().transpose()], ignore_index=True)


        # Display a message box indicating successful data recording
        messagebox.showinfo("Success", f"Data recorded: Patient ID: {patient_id}, Electrode: {electrode_type}, Distance: {distance}, Amperage: {amperage}")

    def get_ai_prediction(self):
        patient_id = self.patient_id_entry.get()
        electrode_type = self.electrode_combobox.get()
    
        # Checking if the selected electrode is in the unique_electrodes list
        if electrode_type in ['Electro A', 'Electro B']:
            # Assuming a hypothetical data point for prediction
            data_point = np.array([[0]]) if electrode_type == 'Electro A' else np.array([[1]])
    
            # Use AI, RNN, and GRU models for prediction
            prediction = self.model.predict(data_point)
            rnn_prediction = self.rnn_model.predict(data_point)
            gru_prediction = self.gru_model.predict(data_point)

    
            # Update the data dataframe with AI, RNN, and GRU predictions
            ai_row = pd.Series({'AI_Distance': prediction[0][0], 'AI_Amperage': prediction[0][1]})
            rnn_row = pd.Series({'RNN_Distance': rnn_prediction[0][0], 'RNN_Amperage': rnn_prediction[0][1]})
            gru_row = pd.Series({'GRU_Distance': gru_prediction[0][0], 'GRU_Amperage': gru_prediction[0][1]})
    
            self.data = pd.concat([self.data, ai_row.to_frame().transpose(), rnn_row.to_frame().transpose(), gru_row.to_frame().transpose()], ignore_index=True)
    
            # Display the AI, RNN, and GRU predictions
            self.result_label.config(text=f"AI Prediction: Distance: {prediction[0][0]}, Amperage: {prediction[0][1]}\n"
                                        f"RNN Prediction: Distance: {rnn_prediction[0][0]}, Amperage: {rnn_prediction[0][1]}\n"
                                        f"GRU Prediction: Distance: {gru_prediction[0][0]}, Amperage: {gru_prediction[0][1]}")
        else:
            messagebox.showwarning("Warning", "Selected Electrode not found in the collected data.")

    def simulate_treatment(self):
        # Mocked virtual planning information
        virtual_planning_info = "Mocked Virtual Planning Info\nReplace with actual simulation details."
        self.virtual_planning_text.insert(tk.END, virtual_planning_info)

    def save_data_and_output(self):
        # Save all information to an Excel file or other suitable format
        output_filename = "output.xlsx"
        self.data.to_excel(output_filename, index=False)
        print(f"Data and output saved successfully to {output_filename}")

    def propose_virtual_planning(self):
        # Extract necessary parameters for virtual planning
        patient_id = self.patient_id_entry.get()
        electrode_type = self.electrode_combobox.get()
        distance = self.distance_slider.get()
        amperage = self.amperage_slider.get()
    
        # Generate virtual planning visualization based on parameters
        virtual_planning_info = self.generate_virtual_planning(patient_id, electrode_type, distance, amperage)
    
        # Display or update the virtual planning text or graphics
        self.virtual_planning_text.delete("1.0", tk.END)  # Clear existing content
        self.virtual_planning_text.insert(tk.END, virtual_planning_info)

    def generate_virtual_planning(self, patient_id, electrode_type, distance, amperage):
        # Assumption: Virtual planning is generated based on some random parameters
        # Replace this with your actual logic for generating virtual planning information
    
        # Mocked logic: Generating random values for virtual planning parameters
        virtual_distance = random.uniform(float(distance) - 2, float(distance) + 2)
        virtual_amperage = random.uniform(float(amperage) - 1, float(amperage) + 1)
    
        virtual_planning_info = f"Virtual Planning: Patient ID - {patient_id}, Electrode - {electrode_type}, Distance - {virtual_distance:.2f}, Amperage - {virtual_amperage:.2f}"
        
        # Add more details based on your specific virtual planning generation logic

        return virtual_planning_info

    def show_scatter_plot(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.data['Distance'], self.data['Amperage'], c='blue', label='Collected Data')
        plt.xlabel('Distance')
        plt.ylabel('Amperage')
        plt.title('Scatter Plot of Collected Data')
        plt.legend()
        plt.show()

    def show_distance_histogram(self):
        plt.figure(figsize=(8, 6))
        plt.hist(self.data['Distance'], bins=20, color='green', edgecolor='black')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.title('Histogram of Distance')
        plt.show()

    def show_amperage_histogram(self):
        plt.figure(figsize=(8, 6))
        plt.hist(self.data['Amperage'], bins=20, color='orange', edgecolor='black')
        plt.xlabel('Amperage')
        plt.ylabel('Frequency')
        plt.title('Histogram of Amperage')
        plt.show()

    def compare_distance_chart(self):
        plt.figure(figsize=(8, 6))
        plt.bar(['Actual', 'AI Prediction'], [self.data['Distance'].mean(), self.data['AI_Distance'].mean()], color=['blue', 'red'])
        plt.xlabel('Data Type')
        plt.ylabel('Mean Distance')
        plt.title('Comparison of Mean Distance - Actual vs AI Prediction')
        plt.show()

    def compare_amperage_chart(self):
        plt.figure(figsize=(8, 6))
        plt.bar(['Actual', 'AI Prediction'], [self.data['Amperage'].mean(), self.data['AI_Amperage'].mean()], color=['orange', 'purple'])
        plt.xlabel('Data Type')
        plt.ylabel('Mean Amperage')
        plt.title('Comparison of Mean Amperage - Actual vs AI Prediction')
        plt.show()



if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectionApp(root)
    root.mainloop() 