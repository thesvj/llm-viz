/** URL prefix when the site is served from a sub-path (e.g. GitHub Pages at /llm-viz). Empty locally. */
export const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

/** Absolute URL of a file in public/. */
export function assetUrl(path: string) {
    return basePath + '/' + path.replace(/^\//, '');
}
