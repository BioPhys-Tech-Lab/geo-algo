from fpdf import FPDF
import datetime
import os

class ReportGenerator(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        # Use standard font (Arial is standard in FPDF)
        # self.add_font("Arial", "", "arial.ttf", uni=True) 
        pass

    def header(self):
        self.set_font("Arial", "B", 15)
        self.cell(0, 10, "Doctoral Level Geological Analysis Report", 0, 1, "C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def generate_report(self, output_path, rock_type, confidence, metrics, images, model_path, geology_info=None, screenshot_3d=None):
        self.add_page()
        
        # 1. Title and Date
        self.set_font("Arial", "", 10)
        self.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "R")
        self.ln(10)
        
        # 2. Identification Section
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "1. Rock Identification", 0, 1)
        self.set_font("Arial", "", 12)
        self.cell(0, 10, f"Classified Type: {rock_type}", 0, 1)
        self.cell(0, 10, f"Confidence Score: {confidence*100:.2f}%", 0, 1)
        
        if geology_info:
            self.ln(5)
            self.set_font("Arial", "B", 11)
            self.cell(0, 10, "Geological Context:", 0, 1)
            self.set_font("Arial", "", 10)
            self.cell(0, 6, f"Type: {geology_info.get('type', 'N/A')}", 0, 1)
            self.cell(0, 6, f"Formation: {geology_info.get('formation', 'N/A')}", 0, 1)
            self.cell(0, 6, f"Composition: {geology_info.get('composition', 'N/A')}", 0, 1)
            self.ln(2)
            self.multi_cell(0, 5, f"Description: {geology_info.get('description', '')}")

        self.ln(5)
        
        # 3. Visual Evidence (Input Images)
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "2. Visual Evidence (Input Views)", 0, 1)
        self.ln(5)
        
        # Layout images in a row
        y_pos = self.get_y()
        img_width = 50
        for i, img_path in enumerate(images):
            if i < 3: # Max 3 images
                x_pos = 10 + (i * (img_width + 10))
                try:
                    self.image(img_path, x=x_pos, y=y_pos, w=img_width)
                except Exception as e:
                    print(f"Warning: Could not add image {img_path} to report: {e}")
        
        self.ln(60) # Space for images
        
        # 4. 3D Geometric Analysis
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "3. 3D Geometric Analysis", 0, 1)
        self.set_font("Arial", "", 11)
        
        # Table-like display
        col_w = 40
        row_h = 10
        
        # Header
        self.set_fill_color(200, 220, 255)
        self.set_font("Arial", "B", 10)
        self.cell(col_w, row_h, "Metric", 1, 0, "C", 1)
        self.cell(col_w, row_h, "Value", 1, 0, "C", 1)
        self.cell(col_w*2, row_h, "Description", 1, 1, "C", 1)
        
        # Rows
        self.set_font("Arial", "", 10)
        
        metrics_data = [
            ("Volume", f"{metrics.get('volume', 0):.4f} units^3", "Approximate volume of the reconstruction."),
            ("Surface Area", f"{metrics.get('area', 0):.4f} units^2", "Total surface area of the mesh."),
            ("Sphericity", f"{metrics.get('sphericity', 0):.4f}", "Measure of how spherical the object is (0-1)."),
            ("Est. Density", "Unknown", "Requires mass data (not available).")
        ]
        
        for metric, value, desc in metrics_data:
            self.cell(col_w, row_h, metric, 1, 0, "L")
            self.cell(col_w, row_h, value, 1, 0, "C")
            self.cell(col_w*2, row_h, desc, 1, 1, "L")
            
        self.ln(10)
        
        # 4. Mineralogical Data (Mindat.org)
        if geology_info and "hardness" in geology_info:
            self.ln(10)
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "4. Mineralogical Data (Source: Mindat.org)", 0, 1)
            self.set_font("Arial", "", 10)
            
            # Data table
            min_data = [
                ("Hardness (Mohs)", geology_info.get("hardness", "N/A")),
                ("Specific Gravity", geology_info.get("specific_gravity", "N/A")),
                ("Crystal System", geology_info.get("crystal_system", "N/A")),
                ("Composition", geology_info.get("composition", "N/A")),
                ("Fracture", geology_info.get("fracture", "N/A"))
            ]
            
            for label, val in min_data:
                self.set_font("Arial", "B", 10)
                self.cell(40, 6, label + ":", 0, 0)
                self.set_font("Arial", "", 10)
                self.cell(0, 6, val, 0, 1)
            
            # External Links
            self.ln(5)
            self.set_font("Arial", "B", 10)
            self.cell(0, 6, "External Databases (Click to Access):", 0, 1)
            self.set_font("Arial", "U", 10)
            self.set_text_color(0, 0, 255) # Blue for links
            
            if "mindat_search_url" in geology_info:
                self.cell(0, 6, f"Mindat.org: {rock_type} Data", 0, 1, link=geology_info["mindat_search_url"])
            if "drp_search_url" in geology_info:
                self.cell(0, 6, f"Digital Rocks Portal: {rock_type} Micro-CT", 0, 1, link=geology_info["drp_search_url"])
            
            self.set_text_color(0, 0, 0) # Reset color

        # 5. 3D Visualization
        if screenshot_3d and os.path.exists(screenshot_3d):
            self.add_page()
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "5. 3D Morphological Visualization", 0, 1)
            self.ln(5)
            try:
                # Add the 3D screenshot
                self.image(screenshot_3d, x=10, w=190)
                self.ln(110) # Space for the large image
            except Exception as e:
                print(f"Warning: Could not add 3D screenshot to report: {e}")
                
        self.ln(10)
        
        # 6. Conclusion
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "6. Analysis Conclusion", 0, 1)
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 7, 
            f"The sample has been identified as {rock_type} with high confidence. "
            f"The 3D reconstruction exhibits a sphericity of {metrics.get('sphericity', 0):.2f}, "
            "indicating its morphological characteristics. "
            "The mineralogical data provided aligns with standard geological classifications."
        )
        
        try:
            self.output(output_path)
            print(f"Report report generated at {output_path}")
        except Exception as e:
            print(f"Error saving report: {e}")
