import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Toplevel, Entry, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import random

class ImageProcessor:
    def __init__(self, root):  
        self.root = root
        self.root.title("🖼 Görüntü İşleme Arayüzü")
        self.root.geometry("1300x900")

        self.image1 = None
        self.image2 = None
        self.processed_image = None
        self.display_size = (200, 200)
        self.rotation_angle = 0  
        self.build_interface()

    def build_interface(self):
        
        bg_image_path ="C:\\Users\\musta\\Desktop\\arkaplan_arayuz.jpg"
        bg_image = Image.open(bg_image_path)
        bg_image = bg_image.resize((1300, 900))  
        self.bg_photo = ImageTk.PhotoImage(bg_image)

        bg_label = Label(self.root, image=self.bg_photo)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        top_frame = Frame(self.root, bg="white")
        top_frame.pack(pady=15)

        Button(top_frame, text="📁 Görsel 1 Yükle", command=self.load_image1,
            bg="#ffca65", fg="white", font=("Arial", 10, "bold"), width=20).grid(row=0, column=0, padx=15)

        Button(top_frame, text="📁 Görsel 2 Yükle", command=self.load_image2,
            bg="#ffca65", fg="white", font=("Arial", 10, "bold"), width=20).grid(row=0, column=1, padx=15)

        image_frame = Frame(self.root, bg="white")
        image_frame.pack(pady=10)

         
        bos_gorsel = Image.new("RGB", (200, 200), color="gray")
        self.bos_gorsel_tk = ImageTk.PhotoImage(bos_gorsel)

          
        self.image1_label = Label(image_frame, image=self.bos_gorsel_tk, bg="gray")
        self.image1_label.grid(row=0, column=0, padx=10)

         
        self.image2_label = Label(image_frame, image=self.bos_gorsel_tk, bg="gray")
        self.image2_label.grid(row=0, column=1, padx=10)

        
        self.result_label = Label(image_frame, image=self.bos_gorsel_tk, bg="gray")
        self.result_label.grid(row=0, column=2, padx=10)

        self.build_button_groups()

    def build_button_groups(self):
        grid_frame = Frame(self.root, bg="#fdf6ec")
        grid_frame.pack(padx=20, pady=10)
        grid_frame.grid_columnconfigure(0, weight=1)
        grid_frame.grid_columnconfigure(1, weight=1)

        groups = [
            ("Temel İşlemler", [
                ("Orijinale Dön", self.reset_to_original),
                ("Gri ", self.apply_gray),
                ("Binary ", self.apply_binary),
                ("90° Sağa Döndürme", self.rotate_right),
                ("90° Sola Döndürme", self.rotate_left),
                (" Kırpma", self.open_crop_window),
                (" Yakınlaştırma", self.zoom_in),
                (" Uzaklaştırma", self.zoom_out)
            ]),
            ("Renk ve Histogram İşlemleri", [
                ("HSV Dönüşümü ", self.color_transform),
                ("Negatif Dönüşüm", self.negative_image),
                (" Gösterim", self.show_histogram),
                (" Germe", self.histogram_germe),
                (" Genişletme", self.histogram_genisletme)
            ]),
            ("Aritmetik İşlemler", [
                ("İki resim arasında ekleme", self.add_images),
                ("İki resim arasında çarpma", self.multiply_images),
                ("Parlaklık Artırma", self.brightness_increase_step),
                ("Parlaklık Azaltma", self.brightness_decrease_step),
            ]),
            ("Filtre, Gürültü ve Temizleme", [
                ("Salt & Pepper", self.add_salt_pepper_noise),
                ("Mean Filtre", self.mean_filter),
                ("Median Filtre", self.median_filter),
                ("Blurring", self.blur_image),
                ("Gauss (Konvolüsyon)", self.gaussian_filter)
            ]),
            ("Kenar ve Eşikleme", [
                ("Sobel Kenar", self.sobel_edge_detection),
                ("Adaptif Eşikle", self.adaptive_threshold)
            ]),
            ("Morfolojik İşlemler", [
                ("Genişleme", self.dilate),
                ("Aşınma", self.erode),
                ("Açma", self.opening),
                ("Kapama", self.closing)
            ])
        ]

       
        for idx, (title, btns) in enumerate(groups):
            frame = Frame(grid_frame, bg="#e6f7ff", width=450, height=50)
            frame.grid_propagate(False)
            frame.grid(row=idx//2, column=idx%2, padx=0, pady=10, sticky="nsew")
            self.group_buttons(title, btns, parent=frame)

    def group_buttons(self, title, buttons, parent=None):
        colors = {
            "Temel İşlemler": "#e3f2fd",
            "Renk ve Histogram İşlemleri": "#e3f2fd",
            "Aritmetik İşlemler": "#f0f4c3",
            "Filtre, Gürültü ve Temizleme": "#f0f4c3",
            "Kenar ve Eşikleme": "#ffe0b2",
            "Morfolojik İşlemler": "#ffe0b2"
        }
        bg_color = colors.get(title, "#ffffff")

        frame = Frame(parent or self.root, bg=bg_color, padx=12, pady=12, relief="ridge", bd=2, width=500)
        frame.pack(pady=5, padx=30, fill="both", expand=True)
        Label(frame, text=f"🔹 {title}", font=("Helvetica", 9, "bold"), fg="black", bg=bg_color, width=25, height=1).pack(anchor="center", pady=(0, 8))

        button_frame = Frame(frame, bg=bg_color)
        button_frame.pack(fill="both", expand=True)

        for i, (text, cmd) in enumerate(buttons):
            btn = Button(button_frame, text=text[:25], command=cmd, bg="#fff", activebackground="#f8c8dc", relief="raised", font=("Verdana", 8), width=25, height=1)
            btn.grid(row=i // 3, column=i % 3, padx=5, pady=5, sticky="nsew")
            button_frame.grid_rowconfigure(i // 3, weight=1)
            button_frame.grid_columnconfigure(i % 3, weight=1)

    def load_image1(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image1 = Image.open(file_path).convert("RGB")
            self.display_image(self.image1, self.image1_label)

    def load_image2(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image2 = Image.open(file_path).convert("RGB")
            self.display_image(self.image2, self.image2_label)

    def display_image(self, img, label):
        img = img.copy()
        img.thumbnail(self.display_size, Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        label.config(image=tk_img, width=self.display_size[0], height=self.display_size[1])
        label.image = tk_img

    def update_result(self):
        if self.processed_image:
            self.display_image(self.processed_image, self.result_label)

    def reset_to_original(self):
        if self.image1:
            self.processed_image = self.image1.copy()
            self.update_result()

    def apply_gray(self):
        if self.image1:
            img = np.array(self.image1)
       
            gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        
            gray_rgb = np.stack((gray,) * 3, axis=-1).astype(np.uint8)
            self.processed_image = Image.fromarray(gray_rgb)
            self.update_result()

    def blur_image(self):
        if self.image1:
            img = np.array(self.image1)
            result = np.copy(img)
            kernel = np.ones((3, 3)) / 9  

        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                for c in range(3):  
                    region = img[i-1:i+2, j-1:j+2, c]
                    result[i, j, c] = np.sum(region * kernel)
        self.processed_image = Image.fromarray(result.astype(np.uint8))
        self.update_result()

    def apply_binary(self):
        if self.image1:
            img = np.array(self.image1)
            gray = np.mean(img, axis=2).astype(np.uint8)
            binary = ((gray > 128) * 255).astype(np.uint8)
            self.processed_image = Image.fromarray(np.stack([binary]*3, axis=-1))
            self.update_result()

    def rotate_right(self):
        if self.image1:
            
            self.rotation_angle = (self.rotation_angle + 90) % 360
            self.processed_image = self.image1.rotate(-self.rotation_angle, expand=True)
            self.update_result()

    def rotate_left(self):
        if self.image1:
            
            self.rotation_angle = (self.rotation_angle - 90) % 360
            self.processed_image = self.image1.rotate(-self.rotation_angle, expand=True)
            self.update_result()

    def open_crop_window(self):
        if not self.image1:
            return

        crop_win = Toplevel(self.root)
        crop_win.title("Kırpma Değerleri")
        crop_win.geometry("250x200")

        entries = {}
        for i, label in enumerate(["Üst", "Sağ", "Alt", "Sol"]):
            tk.Label(crop_win, text=f"{label} (px):").grid(row=i, column=0, padx=10, pady=5, sticky="w")
            entry = tk.Entry(crop_win)
            entry.grid(row=i, column=1, padx=10, pady=5)
            entries[label.lower()] = entry

        def apply_crop():
            try:
                top = int(entries["üst"].get())
                right = int(entries["sağ"].get())
                bottom = int(entries["alt"].get())
                left = int(entries["sol"].get())

                w, h = self.image1.size
                self.processed_image = self.image1.crop((left, top, w - right, h - bottom))
                self.update_result()
                crop_win.destroy()
            except Exception as e:
                messagebox.showerror("Hata", f"Geçerli sayılar girin!\n{e}")

        tk.Button(crop_win, text="Kırp", command=apply_crop, bg="lightblue").grid(row=4, columnspan=2, pady=10)

    def zoom_in(self):
        if self.image1:
           
            w, h = self.processed_image.size if self.processed_image else self.image1.size
           
            new_size = (int(w * 1.1), int(h * 1.1))
            self.processed_image = self.image1.resize(new_size, Image.Resampling.LANCZOS)
            self.update_result()

    def zoom_out(self):
        if self.image1:
            
            w, h = self.processed_image.size if self.processed_image else self.image1.size
          
            new_size = (int(w * 0.9), int(h * 0.9))
            self.processed_image = self.image1.resize(new_size, Image.Resampling.LANCZOS)
            self.update_result()

    def color_transform(self):
        if self.image1:
            img = np.array(self.image1).astype(float)
        hsv = np.zeros_like(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                r = img[i, j, 0] / 255.0
                g = img[i, j, 1] / 255.0
                b = img[i, j, 2] / 255.0

                max_val = max(r, g, b)
                min_val = min(r, g, b)
                delta = max_val - min_val

                if delta == 0:
                    h = 0
                elif max_val == r:
                    h = (60 * ((g - b) / delta)) % 360
                elif max_val == g:
                    h = (60 * ((b - r) / delta)) + 120
                else: 
                    h = (60 * ((r - g) / delta)) + 240

               
                s = 0 if max_val == 0 else delta / max_val

              
                v = max_val

                hsv[i, j, 0] = int(h / 360 * 255)
                hsv[i, j, 1] = int(s * 255)
                hsv[i, j, 2] = int(v * 255)

        self.processed_image = Image.fromarray(hsv.astype(np.uint8))
        self.update_result()

    def negative_image(self):
        if self.image1:
            img = np.array(self.image1)
            negative = 255 - img
            self.processed_image = Image.fromarray(negative.astype(np.uint8))
            self.update_result()    

    def show_histogram(self):
        if self.image1:
            gray_image = self.image1.convert("L")
            histogram = gray_image.histogram()

            width, height = 256, 100
            hist_img = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(hist_img)

            max_value = max(histogram)
            scaled_hist = [int(h / max_value * height) for h in histogram]

            for x in range(256):
                draw.line((x, height, x, height - scaled_hist[x]), fill="black")

            self.processed_image = hist_img
            self.update_result()

    def histogram_germe(self):
        if self.image1:
            gray_image = self.image1.convert("L")
            img_np = np.array(gray_image)

            min_val = np.min(img_np)
            max_val = np.max(img_np)

            if max_val == min_val:
                stretched = img_np.copy()
            else:
                stretched = ((img_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)

            self.processed_image = Image.fromarray(stretched)
            self.update_result()

    def histogram_genisletme(self):
        if self.image1:
            gray_image = self.image1.convert("L")
            img_np = np.array(gray_image)

            hist, bins = np.histogram(img_np.flatten(), 256, [0,256])
            cdf = hist.cumsum()
            cdf_masked = np.ma.masked_equal(cdf, 0)

            cdf_min = cdf_masked.min()
            cdf_max = cdf_masked.max()

            if cdf_max == cdf_min:
                equalized = img_np.copy()
            else:
                cdf_normalized = (cdf_masked - cdf_min) * 255 / (cdf_max - cdf_min)
                cdf_final = np.ma.filled(cdf_normalized, 0).astype(np.uint8)
                equalized = cdf_final[img_np]

            self.processed_image = Image.fromarray(equalized)
            self.update_result()

    def add_images(self):
        if self.image1 and self.image2:
            img1 = self.image1.resize(self.image2.size)
            img2 = self.image2
            arr1 = np.array(img1, dtype=np.int16)
            arr2 = np.array(img2, dtype=np.int16)
            added = np.clip(arr1 + arr2, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(added)
            self.update_result()

    def multiply_images(self):
        if self.image1 and self.image2:
            img1 = self.image1.resize(self.image2.size)
            img2 = self.image2
            arr1 = np.array(img1, dtype=np.float32) / 255.0
            arr2 = np.array(img2, dtype=np.float32) / 255.0
            multiplied = np.clip(arr1 * arr2, 0, 1.0)
            result = (multiplied * 255).astype(np.uint8)
            self.processed_image = Image.fromarray(result)
            self.update_result()

    def brightness_increase_step(self):
        if self.image1:
            img = self.processed_image if self.processed_image else self.image1
            arr = np.array(img, dtype=np.int16)
            arr = np.clip(arr + 10, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(arr)
            self.update_result()

    def brightness_decrease_step(self):
        if self.image1:
            img = self.processed_image if self.processed_image else self.image1
            arr = np.array(img, dtype=np.int16)
            arr = np.clip(arr - 10, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(arr)
            self.update_result()

    def add_salt_pepper_noise(self):
        if self.image1:
            img = np.array(self.image1)
            amount = 0.05
            num_noise = int(img.shape[0] * img.shape[1] * amount)
            for _ in range(num_noise):
                x = random.randint(0, img.shape[0] - 1)
                y = random.randint(0, img.shape[1] - 1)
                img[x, y] = 0 if random.random() < 0.5 else 255
            self.processed_image = Image.fromarray(img)
            self.update_result()

    def mean_filter(self):
        if self.image1:
            img = np.array(self.image1)
            result = np.copy(img)
            for i in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    for c in range(3):
                        result[i, j, c] = np.mean(img[i-1:i+2, j-1:j+2, c])
            self.processed_image = Image.fromarray(result)
            self.update_result()

    def median_filter(self):
        if self.image1:
            img = np.array(self.image1)
            result = np.copy(img)
            for i in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    for c in range(3):
                        result[i, j, c] = np.median(img[i-1:i+2, j-1:j+2, c])
            self.processed_image = Image.fromarray(result)
            self.update_result()

    def blur_image(self):
        if self.processed_image:
            img = self.processed_image
        elif self.image1:
            img = self.image1
        else:
            return

        img_array = np.array(img).astype(np.float32)
        kernel = np.ones((3, 3)) / 9 

        padded_img = np.pad(img_array, ((1, 1), (1, 1), (0, 0)), mode='edge')
        blurred = np.zeros_like(img_array)

        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                for c in range(3):
                    region = padded_img[i:i+3, j:j+3, c]
                    blurred[i, j, c] = np.sum(region * kernel)

        blurred = np.clip(blurred, 0, 255).astype(np.uint8)
        self.processed_image = Image.fromarray(blurred)
        self.update_result()

    def gaussian_filter(self):
        img = self.processed_image if self.processed_image else self.image1
        if img:
            arr = np.array(img)
            result = np.copy(arr)

            kernel = np.array([[1, 4, 6, 4, 1],
                               [4, 16, 24, 16, 4],
                               [6, 24, 36, 24, 6],
                               [4, 16, 24, 16, 4],
                               [1, 4, 6, 4, 1]]) / 256.0

            padded = np.pad(arr, ((2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)

            for _ in range(2):  
                for i in range(2, arr.shape[0] - 2):
                    for j in range(2, arr.shape[1] - 2):
                        for c in range(3):
                            region = padded[i-2:i+3, j-2:j+3, c]
                            result[i, j, c] = np.sum(region * kernel)

            result = np.clip(result, 0, 255)
            self.processed_image = Image.fromarray(result.astype(np.uint8))
            self.update_result()

    def sobel_edge_detection(self):
        if self.image1:
            img = np.array(self.image1.convert("L")).astype(float)  

            Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])

            Ky = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])

            h, w = img.shape
            result = np.zeros((h - 2, w - 2))

            for i in range(h - 2):
                for j in range(w - 2):
                    region = img[i:i + 3, j:j + 3]
                    sx = np.sum(Kx * region)
                    sy = np.sum(Ky * region)
                
                    magnitude = (sx ** 2 + sy ** 2)
                    if magnitude < 0 or np.isnan(magnitude):
                        magnitude = 0
                    result[i, j] = min(255, int(magnitude ** 0.5))

      
            edge_image = np.stack((result,) * 3, axis=-1).astype(np.uint8)
            self.processed_image = Image.fromarray(edge_image)
            self.update_result()

    def adaptive_threshold(self):
        if self.image1:
            gray = np.mean(np.array(self.image1), axis=2).astype(np.uint8)
        height, width = gray.shape
        binary = np.zeros_like(gray)

        window_size = 15  
        offset = window_size // 2

        for i in range(height):
            for j in range(width):
               
                top = max(i - offset, 0)
                bottom = min(i + offset + 1, height)
                left = max(j - offset, 0)
                right = min(j + offset + 1, width)

                local_window = gray[top:bottom, left:right]
                local_thresh = np.mean(local_window)

                binary[i, j] = 255 if gray[i, j] > local_thresh else 0

        self.processed_image = Image.fromarray(np.stack([binary]*3, axis=-1).astype(np.uint8))
        self.update_result()

    def dilate(self):
        if self.image1:
            img = np.array(self.image1.convert("L"))  
            kernel = np.ones((3, 3), np.uint8) 
            result = np.zeros_like(img)
            for i in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    region = img[i-1:i+2, j-1:j+2]  
                    result[i, j] = np.max(region)  
            self.processed_image = Image.fromarray(result)
            self.update_result()

    def erode(self):
        if self.image1:
            img = np.array(self.image1.convert("L"))  
            kernel = np.ones((3, 3), np.uint8)  
            result = np.zeros_like(img)
            for i in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    region = img[i-1:i+2, j-1:j+2]  
                    result[i, j] = np.min(region)  
            self.processed_image = Image.fromarray(result)
            self.update_result()

    def opening(self):
        if self.image1:
            img = np.array(self.image1.convert("L"))  
            
            eroded = np.zeros_like(img)
            for i in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    region = img[i-1:i+2, j-1:j+2]  
                    eroded[i, j] = np.min(region)  

        
            dilated = np.zeros_like(eroded)
            for i in range(1, eroded.shape[0] - 1):
                for j in range(1, eroded.shape[1] - 1):
                    region = eroded[i-1:i+2, j-1:j+2]  
                    dilated[i, j] = np.max(region)  

            self.processed_image = Image.fromarray(dilated)
            self.update_result()

    def closing(self):
        if self.image1:
            img = np.array(self.image1.convert("L"))  
            
            dilated = np.zeros_like(img)
            for i in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    region = img[i-1:i+2, j-1:j+2] 
                    dilated[i, j] = np.max(region)  

            
            eroded = np.zeros_like(dilated)
            for i in range(1, dilated.shape[0] - 1):
                for j in range(1, dilated.shape[1] - 1):
                    region = dilated[i-1:i+2, j-1:j+2]  
                    eroded[i, j] = np.min(region)  
            self.processed_image = Image.fromarray(eroded)
            self.update_result()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()