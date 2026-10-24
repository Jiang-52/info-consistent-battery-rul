
import pymupdf
d = pymupdf.open('paper/info_consistent_manuscript/main.pdf')
d[9].get_pixmap(dpi=85).save('paper/info_consistent_manuscript/_final_p10.png')
print(d.page_count)
