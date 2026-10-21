import React from 'react';
import clsx from 'clsx';
import { useProgramState } from '../Sidebar';
import { IProgramState } from '../Program';
import { IModernLayout } from '../arch/ModernLayout';
import { IModernRuntime, setModernPrompt, tempProbs } from '../arch/ModernRuntime';
import { modelModeInfo } from '../arch/ModernModels';
import { IFlowStep } from './FlowSteps';
import { expertsUsed, flowRuntime, flowSteps, goToStep } from './flowState';

const letter = (id: number) => String.fromCharCode(65 + id);
const pct = (p: number) => `${(p * 100).toFixed(p > 0.995 || p < 0.005 ? 0 : 1)}%`;

/** Left-hand guide for the Llama-style and MoE models: the steps of the forward pass, in order. */
export const FlowGuide: React.FC = () => {
    let state = useProgramState();
    let rt = flowRuntime(state);
    let steps = flowSteps(state);
    if (!rt || steps.length === 0) {
        return <div className='p-4 text-gray-500'>Loading model…</div>;
    }
    let layout = state.layout as IModernLayout;
    let idx = Math.min(state.flow.step, steps.length - 1);
    let step = steps[idx];
    let info = modelModeInfo[state.modelMode];
    let groups: { name: string, items: [IFlowStep, number][] }[] = [];
    steps.forEach((s, i) => {
        let g = groups[groups.length - 1];
        if (!g || g.name !== s.group) groups.push(g = { name: s.group, items: [] });
        g.items.push([s, i]);
    });

    return <div className='flex flex-col h-full min-h-0 text-[15px] text-gray-900'>
        {/* header: same shape as the GPT-2 chapter controls */}
        <div className='flex items-center gap-2 pb-2 border-b border-gray-200'>
            <button className='h-8 w-12 rounded border border-gray-300 bg-white hover:bg-gray-50 disabled:opacity-40' disabled={idx === 0} onClick={() => goToStep(state, idx - 1)} title='Previous step (P or [)'>‹</button>
            <div className='flex-1 text-center'>
                <div className='font-semibold'>{info.title}</div>
                <div className='text-xs text-gray-500'>{info.sub}</div>
            </div>
            <button className='h-8 w-12 rounded border border-gray-300 bg-white hover:bg-gray-50 disabled:opacity-40' disabled={idx === steps.length - 1} onClick={() => goToStep(state, idx + 1)} title='Next step (Space, N or ])'>›</button>
        </div>

        <div className='flex flex-1 min-h-0 gap-4 pt-3'>
            {/* the step list doubles as the map of where you are */}
            <nav className='w-52 shrink-0 overflow-y-auto pr-1 text-sm'>
                {groups.map(g => <div key={g.name} className='mb-3'>
                    <div className='text-[11px] font-semibold uppercase tracking-wide text-gray-500 mb-1'>{g.name}</div>
                    {g.items.map(([s, i]) => <div key={s.id} onClick={() => goToStep(state, i)}
                        className={clsx('px-2 py-1 rounded cursor-pointer flex gap-2', i === idx ? 'bg-blue-600 text-white' : 'hover:bg-gray-100 text-gray-700')}>
                        <span className={clsx('tabular-nums w-5 text-right', i === idx ? 'text-blue-100' : 'text-gray-400')}>{i + 1}</span>
                        <span>{s.title}</span>
                    </div>)}
                </div>)}
                <LayerPicker state={state} layout={layout} />
            </nav>

            {/* the current step */}
            <article className='flex-1 min-w-0 overflow-y-auto pr-2'>
                <div className='text-[11px] font-semibold uppercase tracking-wide text-gray-500'>{step.group} · step {idx + 1} of {steps.length}</div>
                <h2 className='text-xl font-semibold mt-0.5 mb-1'>{step.title}</h2>
                {step.shape && <div className='inline-block text-xs font-mono bg-gray-100 text-gray-700 rounded px-1.5 py-0.5 mb-2'>{step.shape}</div>}
                {step.text.map((t, i) => <p key={i} className='mb-2 leading-relaxed'>{t}</p>)}
                {step.vsGpt2 && <div className='my-3 border-l-4 border-blue-300 bg-blue-50 px-3 py-2 text-sm'>
                    <div className='font-semibold text-blue-900 text-xs uppercase tracking-wide mb-0.5'>Compared with GPT-2</div>
                    {step.vsGpt2}
                </div>}
                {step.newer && <div className='my-3 border-l-4 border-amber-300 bg-amber-50 px-3 py-2 text-sm'>
                    <div className='font-semibold text-amber-900 text-xs uppercase tracking-wide mb-0.5'>In newer models</div>
                    {step.newer}
                </div>}
                {step.panel && <div className='mt-3'><Panel state={state} rt={rt} layout={layout} step={step} /></div>}
                <div className='mt-4 pt-2 border-t border-gray-100 text-xs text-gray-500'>
                    Hover any block to see how one number is computed. Keys: <Kbd>Space</Kbd> or <Kbd>N</Kbd> next, <Kbd>P</Kbd> back.
                </div>
            </article>
        </div>
    </div>;
};

const Kbd: React.FC<{ children: React.ReactNode }> = ({ children }) =>
    <span className='inline-block px-1 rounded border border-gray-300 bg-gray-50 font-mono text-[11px] text-gray-700'>{children}</span>;

const LayerPicker: React.FC<{ state: IProgramState, layout: IModernLayout }> = ({ state, layout }) => {
    return <div className='mt-2 pt-2 border-t border-gray-100'>
        <div className='text-[11px] font-semibold uppercase tracking-wide text-gray-500 mb-1'>Walk through layer</div>
        <div className='flex gap-1'>
            {layout.blocks.map((_, l) => <button key={l} onClick={() => { state.flow.layer = l; goToStep(state, state.flow.step); }}
                className={clsx('w-8 h-7 rounded border text-sm', l === state.flow.layer ? 'bg-blue-600 text-white border-blue-600' : 'bg-white border-gray-300 hover:bg-gray-50')}>{l + 1}</button>)}
        </div>
    </div>;
};

/** Token chips. Clicking one makes the guide follow that position. */
const TokenRow: React.FC<{ state: IProgramState, rt: IModernRuntime, showIds?: boolean, cache?: boolean }> = ({ state, rt, showIds, cache }) => {
    let T = rt.model.config.block_size;
    return <div className='flex flex-wrap gap-1'>
        {Array.from({ length: T }, (_, t) => {
            let real = t < rt.inputLen;
            let followed = t === state.flow.token;
            let isNew = cache && real && t === rt.inputLen - 1 && rt.lastStep !== null;
            return <button key={t} disabled={!real} onClick={() => { state.flow.token = t; state.markDirty(); }}
                title={real ? `Follow position ${t}` : 'Empty slot (not part of the prompt yet)'}
                className={clsx('w-9 rounded border text-center leading-tight py-0.5',
                    !real && 'border-dashed border-gray-300 text-gray-300 cursor-default',
                    real && !followed && 'border-gray-300 bg-white hover:bg-gray-50',
                    followed && 'border-blue-600 ring-2 ring-blue-200 bg-white',
                    isNew && 'bg-amber-50 border-amber-400')}>
                <div className='font-mono font-semibold'>{real ? letter(rt.tokens[t]) : '·'}</div>
                {showIds && <div className='text-[10px] text-gray-500'>{real ? rt.tokens[t] : ''}</div>}
                {cache && real && <div className='text-[9px] text-gray-500'>{isNew ? 'new' : 'cached'}</div>}
            </button>;
        })}
    </div>;
};

const Bar: React.FC<{ p: number, color?: string, label?: string }> = ({ p, color, label }) =>
    <div className='flex items-center gap-2 text-xs'>
        {label && <span className='w-4 font-mono font-semibold'>{label}</span>}
        <div className='flex-1 h-3 bg-gray-100 rounded overflow-hidden'>
            <div className={clsx('h-full', color ?? 'bg-blue-500')} style={{ width: `${Math.max(0, Math.min(1, p)) * 100}%` }} />
        </div>
        <span className='w-12 text-right tabular-nums text-gray-600'>{pct(p)}</span>
    </div>;

const Box: React.FC<{ title: string, children: React.ReactNode }> = ({ title, children }) =>
    <div className='rounded-lg border border-gray-200 bg-white p-3'>
        <div className='text-xs font-semibold text-gray-700 mb-2'>{title}</div>
        {children}
    </div>;

const Panel: React.FC<{ state: IProgramState, rt: IModernRuntime, layout: IModernLayout, step: IFlowStep }> = ({ state, rt, layout, step }) => {
    let c = rt.model.config;
    let T = c.block_size, V = c.vocab_size, nH = c.n_head, nKV = c.n_kv_head, rep = nH / nKV;
    let t = Math.min(state.flow.token, rt.inputLen - 1);
    let l = state.flow.layer;
    let follow = <div className='mb-2'><TokenRow state={state} rt={rt} /></div>;

    switch (step.panel) {
    case 'tokens':
        return <Box title={`Prompt: ${rt.inputLen} tokens, ${T - rt.inputLen} empty slots`}>
            <TokenRow state={state} rt={rt} showIds />
            <div className='text-xs text-gray-500 mt-2'>Dashed slots are empty. The model still computes them, but no real token ever looks at them, so they do not affect the answer.</div>
        </Box>;

    case 'residual': {
        let names = (n: string) => n === 'embed' ? 'After embedding' : n.replace(/l(\d+)\.resid1/, (_, k) => `Layer ${+k + 1}, after attention`).replace(/l(\d+)\.resid2/, (_, k) => `Layer ${+k + 1}, after ${c.n_expert ? 'experts' : 'feed-forward'}`);
        let last = rt.lens[rt.lens.length - 1].probs.data;
        let real = rt.rec['probs'].data;
        let maxDiff = 0;
        for (let i = 0; i < last.length; i++) maxDiff = Math.max(maxDiff, Math.abs(last[i] - real[i]));
        return <Box title={`Logit lens: the guess for the token after position ${t} ("${letter(rt.tokens[t])}")`}>
            {follow}
            <table className='w-full text-xs'>
                <thead><tr className='text-gray-500'><th className='text-left font-normal pb-1'>Stop after…</th>{Array.from({ length: V }, (_, v) => <th key={v} className='font-mono font-semibold pb-1'>{letter(v)}</th>)}<th className='font-normal pb-1'>Guess</th></tr></thead>
                <tbody>{rt.lens.map(({ name, probs }) => {
                    let row = Array.from(probs.data.slice(t * V, t * V + V));
                    let best = row.indexOf(Math.max(...row));
                    return <tr key={name} className='border-t border-gray-100'>
                        <td className='py-1 pr-2 text-gray-700'>{names(name)}</td>
                        {row.map((p, v) => <td key={v} className='px-1'><div className='h-3 bg-gray-100 rounded overflow-hidden'><div className={clsx('h-full', v === best ? 'bg-blue-500' : 'bg-gray-400')} style={{ width: `${p * 100}%` }} /></div></td>)}
                        <td className='text-center font-mono font-semibold'>{letter(best)} <span className='font-normal text-gray-500'>{pct(row[best])}</span></td>
                    </tr>;
                })}</tbody>
            </table>
            <div className='text-xs text-gray-500 mt-2'>
                Each row applies the final RMS norm and LM head early. This is a probe, not part of the model.
                The last row is the real output (matches it {maxDiff < 1e-6 ? 'exactly' : `to ${maxDiff.toExponential(1)}`}).
            </div>
        </Box>;
    }

    case 'gqa': {
        let w = 360, qx = (h: number) => 20 + h * ((w - 40) / Math.max(1, nH - 1));
        let kx = (g: number) => {
            let hs = Array.from({ length: rep }, (_, j) => qx(g * rep + j));
            return (Math.min(...hs) + Math.max(...hs)) / 2;
        };
        return <Box title={`${nH} query heads, ${nKV} shared key/value heads`}>
            <svg viewBox={`0 0 ${w} 110`} className='w-full max-w-md'>
                {Array.from({ length: nH }, (_, h) => <line key={h} x1={qx(h)} y1={30} x2={kx(Math.floor(h / rep))} y2={78} stroke='#94a3b8' strokeWidth={1.5} />)}
                {Array.from({ length: nH }, (_, h) => <g key={h}><rect x={qx(h) - 16} y={10} width={32} height={20} rx={4} fill='#dbeafe' stroke='#3b82f6' /><text x={qx(h)} y={24} textAnchor='middle' fontSize={11}>Q{h}</text></g>)}
                {Array.from({ length: nKV }, (_, g) => <g key={g}><rect x={kx(g) - 30} y={78} width={60} height={22} rx={4} fill='#dcfce7' stroke='#16a34a' /><text x={kx(g)} y={93} textAnchor='middle' fontSize={11}>K{g} · V{g}</text></g>)}
            </svg>
            <div className='text-xs text-gray-500 mt-1'>In the 3D view, each group's query heads sit next to the K and V they share, with a wider gap between groups.</div>
        </Box>;
    }

    case 'rope': {
        let A = c.head_dim ?? c.n_embd / nH, h = A / 2;
        let q = rt.rec[`l${l}.attn.q`].data, qr = rt.rec[`l${l}.attn.q_rope`].data;
        let base = (0 * T + t) * A; // head 0, position t
        return <Box title={`Rotation angles at position ${t} ("${letter(rt.tokens[t])}")`}>
            {follow}
            <div className='flex flex-wrap gap-3'>
                {Array.from({ length: h }, (_, j) => {
                    let ang = t / Math.pow(c.rope_theta, (2 * j) / A);
                    let deg = (ang * 180 / Math.PI) % 360;
                    return <div key={j} className='text-center text-[11px]'>
                        <svg width={44} height={44} viewBox='-22 -22 44 44'>
                            <circle r={18} fill='none' stroke='#e5e7eb' strokeWidth={2} />
                            <line x1={0} y1={0} x2={18} y2={0} stroke='#cbd5e1' strokeWidth={1.5} />
                            <line x1={0} y1={0} x2={18 * Math.cos(ang)} y2={-18 * Math.sin(ang)} stroke='#2563eb' strokeWidth={2.5} strokeLinecap='round' />
                        </svg>
                        <div className='text-gray-500'>pair {j}</div>
                        <div className='tabular-nums'>{deg.toFixed(0)}°</div>
                    </div>;
                })}
            </div>
            <div className='text-xs text-gray-600 mt-2'>
                Pair 0 turns by 1 radian per position, later pairs turn slower (the angle shrinks by a factor {c.rope_theta.toLocaleString('en-US')}<sup>2/{A}</sup> per pair).
                Real numbers for head 0, pair 0: ({q[base].toFixed(3)}, {q[base + h].toFixed(3)}) becomes ({qr[base].toFixed(3)}, {qr[base + h].toFixed(3)}). Its length stays the same.
            </div>
        </Box>;
    }

    case 'mask': {
        let n = rt.inputLen;
        return <Box title='Which earlier tokens each token may look at'>
            <div className='inline-grid gap-px text-[10px]' style={{ gridTemplateColumns: `16px repeat(${n}, 18px)` }}>
                <div />
                {Array.from({ length: n }, (_, k) => <div key={k} className='text-center font-mono text-gray-500'>{letter(rt.tokens[k])}</div>)}
                {Array.from({ length: n }, (_, q) => <React.Fragment key={q}>
                    <div className={clsx('font-mono text-right pr-1', q === t ? 'text-blue-700 font-bold' : 'text-gray-500')}>{letter(rt.tokens[q])}</div>
                    {Array.from({ length: n }, (_, k) => <div key={k} className={clsx('h-[18px] rounded-sm', k <= q ? (q === t ? 'bg-blue-500' : 'bg-blue-200') : 'bg-gray-100')} />)}
                </React.Fragment>)}
            </div>
            <div className='text-xs text-gray-500 mt-2'>Rows are the token doing the looking; columns are the tokens it can see. Grey cells are masked (score −∞, weight 0).</div>
        </Box>;
    }

    case 'attention': {
        let probs = rt.rec[`l${l}.attn.probs`].data;
        let n = rt.inputLen;
        let cellPx = 14;
        return <Box title={`Attention weights in layer ${l + 1} (rows sum to 1)`}>
            {follow}
            <div className='flex flex-wrap gap-4'>
                {Array.from({ length: nH }, (_, h) => <div key={h}>
                    <div className='text-xs text-gray-600 mb-1'>Head {h} <span className='text-gray-400'>· uses K{Math.floor(h / rep)} V{Math.floor(h / rep)}</span></div>
                    <div className='inline-grid gap-px' style={{ gridTemplateColumns: `repeat(${n}, ${cellPx}px)` }}>
                        {Array.from({ length: n * n }, (_, i) => {
                            let q = Math.floor(i / n), k = i % n;
                            let p = k <= q ? probs[(h * T + q) * T + k] : 0;
                            return <div key={i} title={k <= q ? `${letter(rt.tokens[q])}@${q} → ${letter(rt.tokens[k])}@${k}: ${pct(p)}` : 'masked'}
                                style={{ width: cellPx, height: cellPx, background: k <= q ? `rgba(37, 99, 235, ${0.08 + 0.92 * p})` : '#f3f4f6', outline: q === t ? '1px solid #1d4ed8' : undefined }} />;
                        })}
                    </div>
                </div>)}
            </div>
            <div className='text-xs text-gray-500 mt-2'>The outlined row is the followed token. Darker = more weight. The "Attention maps" button on the 3D view shows every layer.</div>
        </Box>;
    }

    case 'moe': {
        let E = c.n_expert!, K = c.top_k!;
        let rp = Array.from(rt.rec[`l${l}.mlp.router_probs`].data.slice(t * E, t * E + E));
        let wts = Array.from(rt.rec[`l${l}.mlp.weights`].data.slice(t * E, t * E + E));
        let used = expertsUsed(state);
        return <Box title={`Layer ${l + 1} router, for position ${t} ("${letter(rt.tokens[t])}")`}>
            {follow}
            <div className='grid grid-cols-[auto_1fr_1fr] gap-x-3 gap-y-1 items-center text-xs'>
                <div /><div className='text-gray-500'>Router probability</div><div className='text-gray-500'>Weight after top-{K}</div>
                {rp.map((p, e) => <React.Fragment key={e}>
                    <div className={clsx('font-semibold', used.includes(e) ? 'text-blue-700' : 'text-gray-400')}>Expert {e}</div>
                    <Bar p={p} color={used.includes(e) ? 'bg-blue-500' : 'bg-gray-300'} />
                    <Bar p={wts[e]} color='bg-amber-500' />
                </React.Fragment>)}
            </div>
            <div className='text-xs text-gray-600 mt-2'>
                This token uses expert {used.join(' and expert ')}: output = {used.map(e => `${wts[e].toFixed(3)} × expert ${e}`).join(' + ')}. The 3D view dims the experts it does not use.
            </div>
        </Box>;
    }

    case 'output': {
        let temp = state.flow.temperature;
        let p1 = tempProbs(rt, t, 1), pT = tempProbs(rt, t, temp);
        return <Box title={`Next-token guess after position ${t} ("${letter(rt.tokens[t])}")`}>
            {follow}
            <label className='flex items-center gap-3 text-sm mb-2'>
                <span>Temperature</span>
                <input type='range' min={0} max={3} step={0.1} value={temp} className='flex-1'
                    onChange={ev => { state.flow.temperature = +ev.target.value; state.markDirty(); }} />
                <span className='w-10 tabular-nums text-right'>{temp.toFixed(1)}</span>
            </label>
            <div className='grid grid-cols-2 gap-4'>
                <div><div className='text-xs text-gray-500 mb-1'>Model output (temperature 1)</div>{p1.map((p, v) => <Bar key={v} p={p} label={letter(v)} color='bg-gray-400' />)}</div>
                <div><div className='text-xs text-gray-500 mb-1'>Used for sampling (temperature {temp.toFixed(1)})</div>{pT.map((p, v) => <Bar key={v} p={p} label={letter(v)} />)}</div>
            </div>
            <div className='text-xs text-gray-500 mt-2'>This small model is very sure of itself, so you need a high temperature to see much change. Temperature 0 always picks the top token.</div>
        </Box>;
    }

    case 'cache': {
        let s = rt.lastStep;
        let full = rt.inputLen >= T;
        return <Box title='Generate one token at a time'>
            <TokenRow state={state} rt={rt} cache />
            <div className='flex gap-2 mt-3'>
                <button disabled={full} onClick={() => { state.stepModel = true; state.flow.token = Math.min(rt.inputLen, T - 1); state.markDirty(); }}
                    className='px-3 h-8 rounded bg-blue-600 text-white disabled:opacity-40 hover:bg-blue-700'>Next token</button>
                <button onClick={() => { setModernPrompt(rt, [2, 1, 0, 1, 1, 2]); state.flow.token = rt.inputLen - 1; state.markDirty(); }}
                    className='px-3 h-8 rounded border border-gray-300 bg-white hover:bg-gray-50'>Reset prompt</button>
                <span className='self-center text-xs text-gray-500'>temperature {state.flow.temperature.toFixed(1)} (set in the previous step)</span>
            </div>
            {s && <div className='text-xs text-gray-700 mt-3 space-y-1'>
                <div>Picked <b className='font-mono'>{letter(s.picked)}</b> for position {s.pos + 1}: the random number {s.draw.toFixed(3)} fell in its share ({s.probs.map((p, v) => `${letter(v)} ${pct(p)}`).join(', ')}).</div>
                <div className={s.cacheSame ? 'text-green-700' : 'text-red-700'}>
                    {s.cacheSame
                        ? `Checked: the keys and values of the ${s.cacheCols} earlier positions are identical to the previous run, in every layer. Only the new column had to be computed.`
                        : 'Warning: earlier keys or values changed. This should not happen with a causal model.'}
                </div>
            </div>}
            {full && <div className='text-xs text-gray-500 mt-2'>All {T} positions are filled. Reset to start again.</div>}
        </Box>;
    }
    }
    return null;
};
