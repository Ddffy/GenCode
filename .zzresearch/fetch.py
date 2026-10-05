import sys, re, urllib.request, html, os
def get(url):
    req=urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36','Accept':'text/html,application/xhtml+xml,*/*'})
    return urllib.request.urlopen(req, timeout=45).read().decode('utf-8','ignore')
def text(h):
    h=re.sub(r'(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>',' ',h)
    h=re.sub(r'(?is)<br\s*/?>','\n',h)
    h=re.sub(r'(?is)</(p|div|li|h1|h2|h3|h4|h5|tr|pre|blockquote)>','\n',h)
    h=re.sub(r'(?s)<[^>]+>',' ',h)
    h=html.unescape(h)
    h=re.sub(r'[ \t\xa0]+',' ',h)
    h=re.sub(r'\n\s*\n+','\n\n',h)
    return h.strip()
for url in sys.argv[1:]:
    name=re.sub(r'[^a-zA-Z0-9]+','_',url)[-90:]
    try:
        t=text(get(url))
        open(name,'w',encoding='utf-8').write('URL: '+url+'\n\n'+t)
        print('OK', len(t), name, url)
    except Exception as e:
        print('FAIL', type(e).__name__, str(e)[:80], url)
