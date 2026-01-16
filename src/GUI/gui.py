import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pathlib import Path
import cv2 as cv
from PIL import Image, ImageTk

def show_directory_selection_window():
    """
    shows a GUI window to select a directory.
    args: None
    Returns: str (directory_path) or None if cancelled
    """
    result = {'directory_path': None, 'cancelled': True}
    
    def on_directory_selected():
        # Open directory dialog
        # determine project root: one level above this src file
        try:
            project_root = Path(__file__).resolve().parent.parent
        except Exception:
            project_root = None

        initial_dir = str(project_root) if project_root and project_root.exists() else os.path.expanduser("~")

        dir_path = filedialog.askdirectory(
            title="Selecteer een map",
            initialdir=initial_dir
        )
        
        if dir_path:
            result['directory_path'] = dir_path
            result['cancelled'] = False
            root.quit()
    
    def on_cancel():
        result['cancelled'] = True
        root.quit()
    
    # Create main window
    root = tk.Tk()
    root.title("Mondriaan Detector - Map Selectie")
    root.geometry("500x400")
    root.configure(bg='#f0f0f0')
    root.resizable(False, False)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (250)
    y = (root.winfo_screenheight() // 2) - (200)
    root.geometry(f"500x400+{x}+{y}")
    
    # Create main frame
    main_frame = ttk.Frame(root, padding="30")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Configure grid weights
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    
    # Title
    title_label = ttk.Label(main_frame, text="Mondriaan Detector", 
                           font=('Arial', 20, 'bold'))
    title_label.grid(row=0, column=0, pady=(0, 10))
    
    # Subtitle
    subtitle_label = ttk.Label(main_frame, text="Selecteer een map", 
                              font=('Arial', 12))
    subtitle_label.grid(row=1, column=0, pady=(0, 30))
    
    # Directory selection option
    directory_frame = ttk.LabelFrame(main_frame, text="Map Selectie", padding="20")
    directory_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 30))
    main_frame.columnconfigure(0, weight=1)
    directory_frame.columnconfigure(0, weight=1)
    
    directory_desc = ttk.Label(directory_frame, text="Selecteer een map\nvan uw computer", 
                              font=('Arial', 10), justify='center')
    directory_desc.grid(row=0, column=0, pady=(0, 15))
    
    directory_button = ttk.Button(directory_frame, text="📁 Selecteer Map", 
                                 command=on_directory_selected,
                                 style='Accent.TButton')
    directory_button.grid(row=1, column=0)
    
    # Cancel button
    cancel_button = ttk.Button(main_frame, text="Annuleren", 
                              command=on_cancel)
    cancel_button.grid(row=3, column=0, pady=(10, 0))
    
    # Handle window close
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    
    # Start the GUI
    root.mainloop()
    root.destroy()
    
    if result['cancelled']:
        return None
    else:
        return result['directory_path']



def show_prediction_window(image, prediction, probability, auto_close_ms=None):
    """
    shows a GUI window with the image, prediction and probability
    Args:
        image: input image (numpy array)
        prediction: predicted label (str)
        probability: probability of the prediction (float)
        auto_close_ms: time in milliseconds to auto-close the window (int or None)
    Returns: None
    """
    # Create main window
    root = tk.Tk()
    root.title("Mondriaan Detector - Resultaat")
    root.geometry("800x400")
    root.configure(bg='#f0f0f0')
    
    # Create main frame
    main_frame = ttk.Frame(root, padding="20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Configure grid weights
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    
    # Title
    title_label = ttk.Label(main_frame, text="Mondriaan Detector", 
                           font=('Arial', 18, 'bold'))
    title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
    
    # Display image (resize for display)
    display_image = cv.resize(image, (300, 225))  # 4:3 aspect ratio
    display_image_rgb = cv.cvtColor(display_image, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(display_image_rgb)
    photo = ImageTk.PhotoImage(pil_image)
    
    image_label = ttk.Label(main_frame, image=photo)
    image_label.grid(row=1, column=0, padx=(0, 20), pady=(0, 20))
    
    # Result frame
    result_frame = ttk.LabelFrame(main_frame, text="Resultaat", padding="15")
    result_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
    
    # Prediction result
    prediction_text = "✓ Dit is een Mondriaan!" 
    if prediction == "mondriaan1":
        prediction_text = "✓ Dit is Mondriaan 1!"
        prediction_color = '#2E8B57'
        confidence_text = f"Zekerheid: {probability * 100:.2f}%"
    elif prediction == "mondriaan2":
        prediction_text = "✓ Dit is Mondriaan 2!"
        prediction_color = '#2E8B57'
        confidence_text = f"Zekerheid: {probability * 100:.2f}%"
    elif prediction == "mondriaan3":
        prediction_text = "✓ Dit is Mondriaan 3!"
        prediction_color = '#2E8B57'
        confidence_text = f"Zekerheid: {probability * 100:.2f}%"
    elif prediction == "mondriaan4":
        prediction_text = "✓ Dit is Mondriaan 4!"
        prediction_color = '#2E8B57'
        confidence_text = f"Zekerheid: {probability * 100:.2f}%"
    elif prediction == "niet_mondriaan":
        prediction_text = "✗ Dit is geen Mondriaan"
        prediction_color = '#DC143C'
        confidence_text = f"Zekerheid: {probability * 100:.2f}%"
    elif prediction == "mondriaan_onbekend":
        prediction_text = "✗ Te lage zekerheid, maak nieuwe foto"
        prediction_color = '#FFA500'
        confidence_text = f"Zekerheid: {probability * 100:.2f}%"

    
    prediction_label = ttk.Label(result_frame, text=prediction_text, 
                                font=('Arial', 14, 'bold'),
                                foreground=prediction_color)
    prediction_label.grid(row=0, column=0, pady=(0, 10))
    
    # Confidence or additional info

    confidence_label = ttk.Label(result_frame, text=confidence_text, 
                                font=('Arial', 10))
    confidence_label.grid(row=1, column=0, pady=(0, 20))
    
    # Close button
    close_button = ttk.Button(result_frame, text="Sluiten", 
                             command=root.destroy,
                             style='Accent.TButton')
    close_button.grid(row=2, column=0, pady=(10, 0))
    
    # Keep reference to photo to prevent garbage collection
    image_label.photo = photo
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Set focus to close button and bind Enter key
    close_button.focus_set()
    root.bind('<Return>', lambda event: root.destroy())
    root.bind('<KP_Enter>', lambda event: root.destroy())
    
    # Auto-close functionality
    if auto_close_ms is not None:
        root.after(auto_close_ms, root.destroy)
    
    # Start the GUI
    root.mainloop()

    return
