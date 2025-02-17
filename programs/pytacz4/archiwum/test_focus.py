import tkinter as tk

class ChoiceWindow:
    def __init__(self, root, title, options, next_window_callback=None):
        self.selected_index = 0
        self.options = options
        self.next_window_callback = next_window_callback
        
        # Create a new window or use root if provided
        self.window = tk.Toplevel(root) if root else tk.Tk()
        self.window.title(title)
        
        # Ensure the window grabs focus
        self.window.grab_set()
        self.window.focus_force()

        # Label for the window
        label = tk.Label(self.window, text=title, font=("Arial", 12))
        label.pack(padx=20, pady=10)

        # Listbox with options
        self.listbox = tk.Listbox(self.window, font=("Arial", 12))
        self.listbox.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        for option in options:
            self.listbox.insert(tk.END, option)
        self.listbox.selection_set(self.selected_index)

        # Handle the selection with keyboard
        self.listbox.bind("<Up>", self.on_key)
        self.listbox.bind("<Down>", self.on_key)
        self.listbox.bind("<Return>", self.on_select)

        # Button to confirm selection (though Enter is bound)
        self.ok_button = tk.Button(self.window, text="OK", command=self.on_select, font=("Arial", 12))
        self.ok_button.pack(pady=10)
        
        # Force focus on the listbox immediately after window appears
        self.window.after(100, self.set_focus)

    def set_focus(self):
        """Set focus to the listbox."""
        self.listbox.focus_set()

    def on_key(self, event):
        """Handle up and down arrow keys."""
        if event.keysym == "Up":
            self.selected_index = max(0, self.selected_index - 1)
        elif event.keysym == "Down":
            self.selected_index = min(len(self.options) - 1, self.selected_index + 1)
        
        # Update the selection in the listbox
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.selected_index)
        self.listbox.activate(self.selected_index)

    def on_select(self, event=None):
        """Handle Enter key or OK button."""
        selected_item = self.listbox.get(self.selected_index)
        print(f"Selected: {selected_item}")
        
        # Close this window
        self.window.destroy()
        
        # If there’s a callback for the next window, show it
        if self.next_window_callback:
            self.next_window_callback()

def show_second_window():
    # Second window options
    second_window_options = ["Red", "Green", "Blue", "Yellow"]
    second_window = ChoiceWindow(root, "Window 2", second_window_options)
    
def show_first_window():
    # First window options
    first_window_options = ["Apple", "Banana", "Cherry", "Date"]
    first_window = ChoiceWindow(root, "Window 1", first_window_options, next_window_callback=show_second_window)

# Main function
root = tk.Tk()
root.withdraw()  # Hide the root window as we only want to show the windows sequentially
show_first_window()

root.mainloop()
