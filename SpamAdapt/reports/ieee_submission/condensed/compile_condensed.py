import os
import io
import time
import tarfile
import certifi
import requests

# Fix SSL bundle environment issue if present
if 'CURL_CA_BUNDLE' in os.environ:
    del os.environ['CURL_CA_BUNDLE']

condensed_dir = os.path.dirname(os.path.abspath(__file__))
tex_file = 'SpamAdapt_IEEE_Condensed.tex'
tex_path = os.path.join(condensed_dir, tex_file)
out_pdf_path = os.path.join(condensed_dir, 'SpamAdapt_IEEE_Condensed.pdf')

files_to_pack = [
    (tex_path, tex_file),
    (os.path.join(condensed_dir, 'fig_performance_comparison.pdf'), 'fig_performance_comparison.pdf'),
    (os.path.join(condensed_dir, 'fig_efficiency_latency.pdf'), 'fig_efficiency_latency.pdf')
]

print('Creating tarball with files:')
tar_stream = io.BytesIO()
with tarfile.open(fileobj=tar_stream, mode='w:gz') as tar:
    for full_path, arc_name in files_to_pack:
        print(f'  Adding {arc_name} ({os.path.getsize(full_path)} bytes)')
        tar.add(full_path, arcname=arc_name)

tar_stream.seek(0)
tar_bytes = tar_stream.read()
print(f'Total archive size: {len(tar_bytes)} bytes')

url = f'https://latexonline.cc/data?target={tex_file}'
files = {'file': ('archive.tar.gz', tar_bytes, 'application/gzip')}

print(f'Submitting compilation request to {url} (timeout=180s)...')
t0 = time.time()
try:
    resp = requests.post(url, files=files, verify=certifi.where(), timeout=180)
    elapsed = time.time() - t0
    print(f'Finished in {elapsed:.1f}s, HTTP status: {resp.status_code}, content size: {len(resp.content)} bytes')
    
    if resp.status_code == 200 and resp.content[:4] == b'%PDF':
        with open(out_pdf_path, 'wb') as f:
            f.write(resp.content)
        print(f'SUCCESS: Compiled PDF saved to: {out_pdf_path}')
        
        # Check page count with pypdf
        import pypdf
        reader = pypdf.PdfReader(out_pdf_path)
        num_pages = len(reader.pages)
        print(f'=== RESULT: Total pages in compiled PDF: {num_pages} ===')
    else:
        print('Compilation failed or returned non-PDF!')
        print('Response header content-type:', resp.headers.get('content-type'))
        print('Response text excerpt:\n', resp.text[:2000])
except Exception as e:
    print('Compilation request exception:', e)
