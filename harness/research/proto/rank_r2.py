import json,re,sys,os,collections
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from patterns import dim_lines,parse
# Neutral topics; polarity comes from the JSON score. ponytail: fixed lexicon, extend when 'other' dominates.
TOPICS={'nonce':r'nonce','capability check':r'capabilit|current_user_can','output escaping':r'escap|esc_(html|attr|url)|xss',
 'input sanitization':r'sanitiz|unslash|\$_(get|post|request|server|cookie)','SQL preparation':r'prepare|interpolat|injection',
 'naming':r'camel ?case|snake_case|naming','docblock':r'docblock|phpdoc|@param|@return|@since',
 'translation':r'translat|text ?domain|\b_[_enx]\(|esc_html__','caching':r'cach|transient','queries in loops':r'n\+1|in a loop|per iteration',
 'error handling':r'error handling|wp_error|exception|return value','accessibility markup':r'aria|\blabel|alt text|screen reader',
 'globals / coupling':r'\bglobal|coupl|singleton','size / responsibilities':r'responsibilit|too long|complex|nested',
 'WP API choice':r'wordpress api|wp api|core function|helper','dependencies':r'dependenc'}
RX={k:re.compile(v,re.I) for k,v in TOPICS.items()}
def rank(responses, lo=4, hi=8, top=10):
    cnt={'anti':collections.Counter(),'pattern':collections.Counter()}; ex={}
    for r in responses:
        j=parse(r) or {}; seen=set()
        for d,_,reason in dim_lines(r.partition('[/REASONING]')[0]):
            s=j.get(d)                                   # JSON is authoritative; absent key = N/A
            if not isinstance(s,int) or lo<s<hi: continue
            kind='anti' if s<=lo else 'pattern'
            for t in [t for t,rx in RX.items() if rx.search(reason)] or ['other']:
                if (kind,d,t) in seen: continue
                seen.add((kind,d,t)); cnt[kind][(d,t)]+=1; ex.setdefault((kind,d,t),reason[:80])
    return {k:[(d,t,n,ex[(k,d,t)]) for (d,t),n in c.most_common(top)] for k,c in cnt.items()}, cnt
if __name__=='__main__':
    R=[json.loads(l)['response'] for f in sys.argv[1:] for l in open(f) if '"response"' in l]
    out,cnt=rank(R)
    for k in out:
        c=cnt[k]; print(f'{k.upper()} total={sum(c.values())} other={sum(v for (d,t),v in c.items() if t=="other")}')
        for d,t,n,e in out[k]: print(f'{n:3} {d:21} {t:22} | {e}')
