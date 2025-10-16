import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from scrapegraphai.graphs import SmartScraperGraph

class ModernScraperGUI:
    def __init__(self, master):
        self.master = master
        master.title("AI Web Scraper")
        master.geometry("1200x800")
        self.setup_style()
        
        # Configure main container
        self.main_frame = ttk.Frame(master)
        self.main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Create widgets
        self.create_input_panel()
        self.create_prompt_editor()
        self.create_output_panel()
        self.create_status_bar()

        # Default configuration
        self.graph_config = {
            "llm": {
                "model": "ollama/deepseek-r1:1.5b",
                "model_tokens": 8192,
                "temperature": 0.0
            },
            "verbose": True,
            "headless": False,
        }

    def setup_style(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('.', background='#f5f5f5')
        self.style.configure('TFrame', background='#f5f5f5')
        self.style.configure('TLabel', 
                           background='#f5f5f5', 
                           foreground='#333333', 
                           font=('Segoe UI', 10))
        self.style.configure('TButton', 
                           foreground='#ffffff', 
                           background='#4CAF50', 
                           font=('Segoe UI', 10, 'bold'), 
                           borderwidth=0)
        self.style.map('TButton', 
                     background=[('active', '#45a049')])
        self.style.configure('TEntry',
                           fieldbackground='#ffffff',
                           borderwidth=1)
        self.style.configure('TCombobox',
                           fieldbackground='#ffffff')

    def create_input_panel(self):
        input_frame = ttk.Frame(self.main_frame)
        input_frame.pack(fill=tk.X, pady=(0, 20))

        # URL Input
        ttk.Label(input_frame, text="Website URL:", font=('Segoe UI', 11)).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.url_entry = ttk.Entry(input_frame, width=60, font=('Segoe UI', 10))
        self.url_entry.grid(row=1, column=0, sticky=tk.EW, padx=(0, 10))
        self.url_entry.insert(0, "https://www.sliit.lk/")

        # Model Input
        ttk.Label(input_frame, text="AI Model:", font=('Segoe UI', 11)).grid(row=0, column=1, sticky=tk.W)
        self.model_entry = ttk.Entry(input_frame, width=25, font=('Segoe UI', 10))
        self.model_entry.grid(row=1, column=1, sticky=tk.EW, padx=(0, 10))
        self.model_entry.insert(0, "ollama/deepseek-r1:1.5b")

        # Scrape Button
        self.scrape_btn = ttk.Button(input_frame, text="Start Scraping", command=self.start_scraping_thread)
        self.scrape_btn.grid(row=1, column=2, sticky=tk.E, padx=(10, 0))

        # Configure grid weights
        input_frame.columnconfigure(0, weight=3)
        input_frame.columnconfigure(1, weight=1)
        input_frame.columnconfigure(2, weight=0)

    def create_prompt_editor(self):
        ttk.Label(self.main_frame, text="Scraping Instructions:", font=('Segoe UI', 11, 'bold')).pack(anchor=tk.W, pady=(5, 0))
        self.prompt_editor = scrolledtext.ScrolledText(
            self.main_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            height=12,
            padx=12,
            pady=8,
            bg='#ffffff',
            relief='flat'
        )
        self.prompt_editor.pack(fill=tk.BOTH, expand=False)
        self.insert_default_prompt()

    def insert_default_prompt(self):
        default_prompt = '''Extract comprehensive information from the webpage including:
- Company/organization description
- Leadership/management details
- Contact information and social media links
- Key services/products offered
- Any notable features or partnerships'''
        self.prompt_editor.insert(tk.END, default_prompt)

    def create_output_panel(self):
        output_frame = ttk.Frame(self.main_frame)
        output_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(output_frame, text="Scraping Results:", font=('Segoe UI', 11, 'bold')).pack(anchor=tk.W)
        self.result_viewer = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            width=120,
            height=25,
            padx=12,
            pady=8,
            bg='#ffffff',
            relief='flat'
        )
        self.result_viewer.pack(fill=tk.BOTH, expand=True)

    def create_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(
            self.master,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=('Segoe UI', 9),
            background='#e0e0e0',
            foreground='#333333'
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def start_scraping_thread(self):
        url = self.url_entry.get().strip()
        prompt = self.prompt_editor.get("1.0", tk.END).strip()
        model = self.model_entry.get().strip()

        if not url:
            messagebox.showerror("Error", "Please enter a valid URL")
            return
        if not model:
            messagebox.showerror("Error", "Please specify an AI model")
            return

        self.scrape_btn.config(state=tk.DISABLED)
        self.status_var.set("Scraping in progress... This may take a few moments")
        self.result_viewer.delete(1.0, tk.END)

        # Update config with selected model
        self.graph_config['llm']['model'] = model

        threading.Thread(
            target=self.run_scraper,
            args=(url, prompt),
            daemon=True
        ).start()

    def run_scraper(self, url, prompt):
        try:
            scraper = SmartScraperGraph(
                prompt=prompt,
                source=url,
                config=self.graph_config
            )
            result = scraper.run()
            error = None
        except Exception as e:
            result = f"Error: {str(e)}"
            error = e

        self.master.after(0, self.show_results, result, error)

    def show_results(self, results, error):
        self.scrape_btn.config(state=tk.NORMAL)
        status_message = "Ready" if not error else "Error occurred during scraping"
        status_color = '#333333'
        
        if error:
            status_color = '#d9534f'
            self.result_viewer.config(fg='#d9534f')
        else:
            self.result_viewer.config(fg='#333333')

        self.status_var.set(status_message)
        self.status_bar.config(foreground=status_color)
        
        self.result_viewer.delete(1.0, tk.END)
        self.result_viewer.insert(tk.END, str(results))

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernScraperGUI(root)
    root.mainloop()