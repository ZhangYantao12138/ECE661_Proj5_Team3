"""Populate the course's three-column PDF template with measured results."""
import io
import json
from pathlib import Path
from html import escape
from reportlab.pdfgen import canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader,PdfWriter,Transformation
from src.paths import workspace_root,delivery_dir

PROJECT=Path(__file__).resolve().parents[1]
WORKSPACE=workspace_root()


def main():
    data=json.loads((PROJECT/'results/publication_data.json').read_text(encoding='utf-8'))
    if data['status']!='complete':raise RuntimeError('Measured final data is required')
    template=WORKSPACE/'work/poster_template/source.pdf'
    if not template.exists():template=PROJECT/'templates/poster_template.pdf'
    if not template.exists():raise RuntimeError('Restore the provided ECE661 poster template first')
    for name,filename in [('Arial','arial.ttf'),('ArialBold','arialbd.ttf')]:pdfmetrics.registerFont(TTFont(name,str(Path(r'C:\Windows\Fonts')/filename)))
    buffer=io.BytesIO();c=canvas.Canvas(buffer,pagesize=(720,405));c.setTitle('ECE661 Team 3: Universal Adversarial Patches')
    mappings={}
    def text(value,x,top,width,height,font=6.3,bold=False):
        style=ParagraphStyle('text',fontName='ArialBold' if bold else 'Arial',fontSize=font,leading=font*1.26,textColor=colors.HexColor('#344854'))
        paragraph=Paragraph(escape(value).replace('\n','<br/>'),style);w,h=paragraph.wrap(width,height)
        if h>height+.01:raise ValueError(f'Poster text overflow at {x},{top}: {h:.2f}>{height}: {value[:70]}')
        paragraph.drawOn(c,x,405-top-h)
    def image(name,x,top,width,height):
        path=PROJECT/'results/figures'/f'{name}.png';pic=ImageReader(str(path));iw,ih=pic.getSize();factor=min(width/iw,height/ih)
        c.drawImage(pic,x+(width-iw*factor)/2,405-top-ih*factor,width=iw*factor,height=ih*factor,mask='auto')
    def find(rows,**keys):return next(r for r in rows if all(r.get(k)==v for k,v in keys.items()))
    def table(headers,rows,x,top,widths):
        style=ParagraphStyle('cell',fontName='Arial',fontSize=5.8,leading=7)
        values=[[Paragraph(escape(str(v)),style) for v in row] for row in [headers]+rows]
        t=Table(values,colWidths=widths)
        t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.5,colors.HexColor('#657b88')),('LINEBELOW',(0,0),(-1,0),.3,colors.HexColor('#9cabb2')),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#657b88')),('LEFTPADDING',(0,0),(-1,-1),2),('RIGHTPADDING',(0,0),(-1,-1),2),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
        w,h=t.wrap(sum(widths),100);t.drawOn(c,x,405-top-h);return h
    text('Universal Adversarial Patches\non CIFAR-10: Transfer and Adaptive Defenses',21.2,14.5,340,40,font=13.8)
    text('Xitai Yu (xy195), Russo Zhang (yz1047)\nBoyi Zhao (bz138), Zijiang Yang (zy232)\nECE661 Team 3 / Fall 2026',382.4,14.5,210,35,font=6)
    text('Motivation and controlled study',21.6,66.1,205,18,font=9,bold=True)
    text('Can one small texture change many images, transfer across classifiers, and survive input defenses? We distinguish learned texture from ordinary occlusion and compare adaptive attacks under explicit budgets.',21.6,91.5,205,48)
    text('1. Four patch sizes, all ten targets, three seeds.\n2. Held-out transfer with step/time comparisons.\n3. Defense-aware attacks and EOT-matched controls.\n561 main patches + 24 strength checks;\n10,000 test images, 8 draws per image.',21.6,147,205,52)
    text('A/B: models and learned texture',21.6,206.7,205,15,font=9,bold=True)
    text('Actual clean test accuracy; 200 epochs per model',21.6,221,205,12,font=6)
    table(['Model','Parameters','Clean test'],[[name,f"{r['parameters']:,}",f"{100*r['clean_accuracy']:.2f}%"] for name,r in zip(['ResNet-20','VGG-11-BN','DenseNet-BC'],data['models'])],21.6,235,[89,58,58])
    image('B_patch_gallery_poster',21,289,205,82)
    text('Real textures, seed 101. Rows: sides 3/5/7/16; columns: U and targets 0-9.',21.6,377,205,14,font=4.6)
    text('C: held-out ensemble transfer',259.9,66.1,205,15,font=9,bold=True)
    text('Same DenseNet; equal steps or measured training time',259.9,80,205,12,font=6)
    image('C_main',259.9,97,205,83)
    fixed=find(data['paired_effects'],part='C',phase='ensemble_steps',size=7,objective='target_macro',comparison='RV - R')
    timed=find(data['paired_effects'],part='C',phase='ensemble_time',size=7,objective='target_macro',comparison='RV - R')
    text(f"RV - R, targeted macro, side 7:\n1000 steps: {fixed['delta_pp']:+.2f} pp [{fixed['ci95_pp'][0]:+.2f}, {fixed['ci95_pp'][1]:+.2f}].\nMatched time: {timed['delta_pp']:+.2f} pp [{timed['ci95_pp'][0]:+.2f}, {timed['ci95_pp'][1]:+.2f}].\n95% intervals resample paired images within classes;\nthree patch seeds retained, classifier weights fixed.",259.9,185,205,51,font=6)
    for x,label in [(260,'Frozen R + V'),(331,'Shared patch'),(402,'Held-out DenseNet')]:
        c.setFillColor(colors.HexColor('#edf2f4'));c.setStrokeColor(colors.HexColor('#738b97'));c.roundRect(x,405-267,61,23,2,fill=1);c.setFont('Arial',5.2);c.setFillColor(colors.HexColor('#344854'));c.drawCentredString(x+30.5,405-258,label)
    for x in [322,393]:c.line(x,405-255,x+7,405-255)
    rows=[]
    for phase in ['ensemble_steps','ensemble_time']:
        for strategy in ['R','V','RV']:
            r=find(data['ensemble'],phase=phase,size=7,strategy=strategy,objective='TASR macro')
            rows.append([('steps' if phase=='ensemble_steps' else 'time')+' / '+strategy,f"{r['mean_steps']:.0f}",f"{100*r['mean']:.2f}%"])
    table(['Budget / source','Mean updates','Target macro'],rows,259.9,281,[84,62,59])
    image('D_main',501.5,66,205,125)
    text('Side 7; UASR and ten-target macro TASR. Error bars: three patch seeds.',501.5,195,205,12,font=4.6)
    text('D: adaptive defense evaluation',501.5,209.9,205,15,font=9,bold=True)
    text('Original K1 / unknown K4 / defense-aware K4',501.5,224,205,12,font=6)
    blur=find(data['paired_effects'],part='D',size=7,defense='gaussian',objective='untargeted',comparison='adaptive K4 - unknown K4')
    resize=find(data['paired_effects'],part='D',size=7,defense='resize_pad',objective='untargeted',comparison='adaptive K4 - unknown K4')
    costs=[r['clean_drop_pp'] for r in data['defenses']]
    text(f"Adaptive - unknown K4, UASR: {blur['delta_pp']:+.2f} pp\n(blur), {resize['delta_pp']:+.2f} pp (resize). Clean loss:\n{costs[0]:.2f} / {costs[1]:.2f} pp. Resize retained the weakest\ncandidate after exceeding its 2 pp validation limit.",501.5,239,205,32,font=5.8)
    image('D_operators',526,274,156,46)
    text('References and limits',501,326,205,12,font=6,bold=True)
    text('Brown et al., Adversarial Patch (2017).\nAthalye et al., Obfuscated Gradients (ICML 2018).\nKrizhevsky, CIFAR-10 (2009). Architecture references,\nall raw results, code, and full details accompany the report.\nDigital fixed-model results; no certified or physical guarantee.\nAI assisted code, execution, analysis, and writing.',501,341,205,48,font=4.8)
    # Faithful crop of the supplied Duke mark. A clipped PDF merge would retain
    # invisible placeholder text in the final content stream and text extraction.
    brand=PROJECT/'templates/duke_brand.png'
    if not brand.exists():raise RuntimeError('Restore the course template Duke brand crop')
    c.drawImage(str(brand),610,345,width=110,height=60,mask='auto')
    c.save();buffer.seek(0)
    writer=PdfWriter();writer.add_page(PdfReader(buffer).pages[0])
    source=WORKSPACE/'work/poster_template/completed_source.pdf'
    source.parent.mkdir(parents=True,exist_ok=True)
    with source.open('wb') as output:writer.write(output)
    final=delivery_dir();final.mkdir(parents=True,exist_ok=True)
    large=PdfWriter();large_page=large.add_blank_page(width=2880,height=1620)
    large_page.merge_transformed_page(PdfReader(str(source)).pages[0],Transformation().scale(4))
    large.add_metadata({'/Title':'ECE661 Team 3: Universal Adversarial Patches','/Author':'ECE661 Team 3','/Subject':'Measured results; course template; 40 x 22.5 inch print canvas'})
    with (final/'ECE661_Poster.pdf').open('wb') as output:large.write(output)
    template_json=WORKSPACE/'work/poster_template/raw-template.json'
    if not template_json.exists():template_json=PROJECT/'templates/poster_template.json'
    template_data=json.loads(template_json.read_text(encoding='utf-8'))['structuredContent']
    for element in template_data['slides'][0]['pageElements']:
        mappings[element['objectId']]='keep protected Duke branding' if element['objectId']=='g19461511ca2_2_38' else 'replace template content with corresponding measured research text, table, plot, or texture'
    (final/'Poster_Template_Mapping.json').write_text(json.dumps(dict(source_id=template_data['presentationId'],template='ECE661 Poster Template',all_element_mapping=mappings,canvas_scale_for_print=4,print_size_inches=[40,22.5],note='All illustrative placeholder figures and text were replaced; no template sample data retained.'),indent=2),encoding='utf-8')
    print(final/'ECE661_Poster.pdf')


if __name__=='__main__':main()
