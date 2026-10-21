import React from 'react';
import { countParams } from '../arch/ArchPresets';
import { IModernConfig } from '../arch/ModernForward';

const fmt = (n: number) => n.toLocaleString('en-US');
const billions = (n: number) => `${(n / 1e9).toFixed(2)} billion`;

/** Left-hand panel while a real (shape-only) Llama-family model is shown. */
export const PresetFacts: React.FC<{ name: string, sub?: string, config: IModernConfig, onBack: () => void }> = ({ name, sub, config: c, onBack }) => {
    let n = countParams(c);
    let A = c.head_dim ?? c.n_embd / c.n_head;
    let rep = c.n_head / c.n_kv_head;
    let rows: [string, string][] = [
        ['Layers', fmt(c.n_layer)],
        ['Numbers per token (model width)', fmt(c.n_embd)],
        ['Query heads / key-value heads', `${c.n_head} / ${c.n_kv_head} (each K/V shared by ${rep})`],
        ['Numbers per head', fmt(A)],
        ['Feed-forward width', fmt(c.n_hidden)],
        ...(c.n_expert ? [['Experts per layer', `${c.n_expert}, each token uses ${c.top_k}`] as [string, string]] : []),
        ['Vocabulary', `${fmt(c.vocab_size)} tokens`],
        ['Parameters', c.n_expert ? `${billions(n.total)} (${billions(n.active)} used per token)` : billions(n.total)],
    ];
    let notes: string[] = [];
    if (c.tied) notes.push(`The output layer reuses the token embedding table, which saves ${Math.round(c.vocab_size * c.n_embd / 1e6)} million parameters.`);
    if (c.qk_norm) notes.push('Each query and key head gets its own RMS norm before RoPE. This keeps attention scores stable in deep models.');
    if (c.sliding_window) notes.push(`Each token attends only to the previous ${fmt(c.sliding_window)} tokens (sliding window), so memory stays bounded for long texts.`);
    if (c.n_expert) notes.push(`Only ${c.top_k} of ${c.n_expert} experts run per token, so it computes like a ${billions(n.active).replace(' billion', 'B')} model but stores ${billions(n.total).replace(' billion', 'B')} parameters.`);

    return <div className='flex flex-col h-full min-h-0 text-[15px] text-gray-900 overflow-y-auto px-3 py-2'>
        <div className='text-[11px] font-semibold uppercase tracking-wide text-gray-500'>Real model · shape only</div>
        <h2 className='text-xl font-semibold mt-0.5'>{name}</h2>
        {sub && <p className='text-sm text-gray-600 mb-3'>{sub}</p>}
        <p className='mb-3 leading-relaxed'>
            Same building blocks as the tiny Llama: RMS norm, RoPE, grouped-query attention and SwiGLU{c.n_expert ? ', with a mixture of experts' : ''}.
            Only the sizes differ. The 3D view shows every matrix at its real shape, without weights, for {fmt(c.block_size)} tokens of context.
        </p>
        <table className='w-full text-sm mb-3'>
            <tbody>{rows.map(([k, v]) => <tr key={k} className='border-t border-gray-100'>
                <td className='py-1.5 pr-3 text-gray-600'>{k}</td>
                <td className='py-1.5 font-medium tabular-nums'>{v}</td>
            </tr>)}</tbody>
        </table>
        {notes.length > 0 && <div className='border-l-4 border-blue-300 bg-blue-50 px-3 py-2 text-sm space-y-1 mb-3'>
            <div className='font-semibold text-blue-900 text-xs uppercase tracking-wide'>What is special</div>
            {notes.map((t, i) => <p key={i}>{t}</p>)}
        </div>}
        <p className='text-xs text-gray-500 mb-3'>Parameter counts are computed from the published config and match the published totals within 1%.</p>
        <div><button onClick={onBack} className='px-3 h-8 rounded bg-blue-600 text-white hover:bg-blue-700'>Back to the step-by-step guide</button></div>
    </div>;
};
