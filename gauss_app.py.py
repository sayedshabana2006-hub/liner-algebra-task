import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext

class GaussianEliminationApp:
    def __init__(self, master):  
        self.master = master
        master.title("Gaussian Elimination Solver")
        
        self.matrix_size = 0
        
        # --- الإعدادات الرئيسية ---
        tk.Label(master, text="عدد المعادلات (N):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.n_entry = tk.Entry(master, width=10)
        self.n_entry.grid(row=0, column=1, padx=5, pady=5)
        
        self.set_size_button = tk.Button(master, text="تحديد الأبعاد", command=self.create_matrix_input)
        self.set_size_button.grid(row=0, column=2, padx=5, pady=5)
        
        # --- حاوية إدخال المصفوفة ---
        self.matrix_frame = tk.Frame(master)
        self.matrix_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10)
        self.matrix_entries = []

        # --- زر الحل ---
        self.solve_button = tk.Button(master, text="حل المعادلات", command=self.solve_system, state=tk.DISABLED)
        self.solve_button.grid(row=2, column=0, columnspan=3, pady=10)

        # --- منطقة الشرح والنتائج ---
        tk.Label(master, text="شرح الخطوات والنتائج:").grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky='w')
        self.output_text = scrolledtext.ScrolledText(master, width=80, height=20, wrap=tk.WORD, font=('Courier', 10))
        self.output_text.grid(row=4, column=0, columnspan=3, padx=10, pady=5)

    def create_matrix_input(self):
        """ينشئ حقول الإدخال للمصفوفة المُعزّزة."""
        try:
            n = int(self.n_entry.get())
            if n <= 0:
                messagebox.showerror("خطأ", "يجب أن يكون عدد المعادلات أكبر من صفر.")
                return
            self.matrix_size = n
            self.solve_button.config(state=tk.NORMAL)
            
            # مسح العناصر القديمة
            for widget in self.matrix_frame.winfo_children():
                widget.destroy()
            self.matrix_entries = []

            # إنشاء حقول الإدخال (n صف × n+1 عمود)
            for i in range(n):
                row_entries = []
                for j in range(n + 1):
                    bg_color = 'lightyellow' if j == n else 'white'
                    e = tk.Entry(self.matrix_frame, width=5, bg=bg_color)
                    e.grid(row=i, column=j, padx=2, pady=2)
                    row_entries.append(e)
                self.matrix_entries.append(row_entries)
                
        except ValueError:
            messagebox.showerror("خطأ", "الرجاء إدخال عدد صحيح لعدد المعادلات.")

    def solve_system(self):
        """ينفذ الخوارزمية ويطبع الشرح."""
        self.output_text.delete(1.0, tk.END)
        
        n = self.matrix_size
        A = np.zeros((n, n + 1))
        
        try:
            for i in range(n):
                for j in range(n + 1):
                    A[i, j] = float(self.matrix_entries[i][j].get())
        except ValueError:
            messagebox.showerror("خطأ", "جميع حقول المصفوفة يجب أن تحتوي على أرقام.")
            return

        self.print_to_output("=============================================")
        self.print_to_output("مصفوفة المدخلات الأصلية:")
        self.print_to_output(str(A))
        self.print_to_output("=============================================\n")
        
        solutions = self._gaussian_elimination_solver(A.copy())

        if solutions is not None:
            self.print_to_output("\n=============================================")
            self.print_to_output("✨ الحلول النهائية للنظام:")
            for i, sol in enumerate(solutions):
                self.print_to_output(f"x{i+1} = {sol:.6f}")
            self.print_to_output("=============================================")

    def print_to_output(self, text):
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)

    def _gaussian_elimination_solver(self, A):
        n = A.shape[0]

        self.print_to_output("--- 📝 بدء عملية التحويل الأمامي (Forward Elimination) ---")
        
        # التحويل الأمامي
        for i in range(n): 
            # Pivoting
            if np.isclose(A[i, i], 0.0):
                self.print_to_output(f"> ⚠ محور الصف {i+1} يساوي صفر. يتم البحث عن صف للتبديل...")
                for k in range(i + 1, n):
                    if not np.isclose(A[k, i], 0.0):
                        A[[i, k]] = A[[k, i]]
                        self.print_to_output(f"> 🔄 تم تبديل الصف {i+1} بالصف {k+1}")
                        break
                else:
                    self.print_to_output("> ❌ لا يوجد حل فريد")
                    return None
            
            # Elimination
            for j in range(i + 1, n):
                if np.isclose(A[j, i], 0.0):
                    continue
                
                factor = A[j, i] / A[i, i]
                self.print_to_output(f"\n> عملية على الصف {j+1}: R{j+1} = R{j+1} - ({factor:.4f}) * R{i+1}")
                
                A[j, :] -= factor * A[i, :]
                self.print_to_output("المصفوفة بعد العملية:")
                self.print_to_output(str(A))

        self.print_to_output("\n--- ✅ التحويل الأمامي انتهى ---")
        self.print_to_output(str(A))
        self.print_to_output("\n--- 🧠 بدء التعويض الخلفي (Back Substitution) ---")

        # التعويض الخلفي
        X = np.zeros(n)

        for i in range(n - 1, -1, -1):
            if np.isclose(A[i, i], 0.0):
                self.print_to_output(f"> 🚨 محور الصف {i+1} صفر. لا يوجد حل.")
                return None

            rhs = A[i, n]
            for j in range(i + 1, n):
                rhs -= A[i, j] * X[j]

            X[i] = rhs / A[i, i]
            self.print_to_output(f"x{i+1} = {X[i]:.6f}")

        return X


# --- تشغيل التطبيق ---
if __name__ == "__main__":  
    root = tk.Tk()
    app = GaussianEliminationApp(root)
    root.mainloop()
