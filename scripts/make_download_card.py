import base64
from pathlib import Path

pptx_path = Path(r"d:\Projects\ESA 4\ATWFS\outputs\presentation\ATWFS_Presentation.pptx")
with open(pptx_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-2 font-sans">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-2xl p-6 shadow-md max-w-xl mx-auto">
    <div class="flex items-center space-x-4 mb-4">
      <div class="w-12 h-12 rounded-xl bg-amber-500/10 text-amber-600 flex items-center justify-center font-bold text-xl border border-amber-500/20">
        PPTX
      </div>
      <div>
        <h3 class="text-base font-bold text-[var(--foreground)]">ATWFS_Presentation.pptx</h3>
        <p class="text-xs text-[var(--muted-foreground)]">14 Slides · 16:9 Widescreen · Fully Editable Presentation</p>
      </div>
    </div>
    
    <div class="bg-[var(--background)] rounded-xl p-4 border border-[var(--border)] mb-5 space-y-2 text-xs">
      <div class="flex justify-between py-1 border-b border-[var(--border)]">
        <span class="text-[var(--muted-foreground)]">Template Style:</span>
        <span class="font-medium text-[var(--foreground)]">Original Hand-Drawn Road & Paper Grid</span>
      </div>
      <div class="flex justify-between py-1 border-b border-[var(--border)]">
        <span class="text-[var(--muted-foreground)]">File Size:</span>
        <span class="font-medium text-[var(--foreground)]">162.5 KB</span>
      </div>
      <div class="flex justify-between py-1">
        <span class="text-[var(--muted-foreground)]">Local Download Path:</span>
        <span class="font-medium text-[var(--foreground)] truncate ml-2">C:\\Users\\varsh\\Downloads\\ATWFS_Presentation.pptx</span>
      </div>
    </div>
    
    <div class="flex items-center space-x-3">
      <a href="data:application/vnd.openxmlformats-officedocument.presentationml.presentation;base64,{b64}" download="ATWFS_Presentation.pptx" class="flex-1 inline-flex items-center justify-center px-5 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-sm transition shadow-sm cursor-pointer">
        <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
        Download Presentation (.pptx)
      </a>
      <a href="file:///C:/Users/varsh/Downloads/ATWFS_Presentation.pptx" class="inline-flex items-center justify-center px-4 py-3 rounded-xl border border-[var(--border)] hover:bg-[var(--secondary)] text-[var(--foreground)] text-sm font-medium transition cursor-pointer">
        Open File
      </a>
    </div>
  </div>
</body>
</html>"""

out_path = Path(r"C:\Users\varsh\.gemini\antigravity\brain\74843c63-b7db-4812-b3c5-7b92e0eceb31\download_card.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_content)
print("Successfully written download_card.html")
