import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

// The page is built into the Python package, which serves it (src/hoi4_arena/live/server.py):
// `npm run build` here, and commit what it writes there.
const page = '../../src/hoi4_arena/live/page';

export default defineConfig({
	// hls.js (~520 kB) is one chunk, loaded only where the browser cannot play HLS itself.
	build: { chunkSizeWarningLimit: 700 },
	plugins: [
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			adapter: adapter({ pages: page, assets: page, fallback: undefined, strict: true }),
			// A fixed name, not the build's time, so the same source builds the same files and
			// CI can check the committed build is current.
			version: { name: 'hoi4-live' },
			prerender: {
				// The app's manifest and icons are drawn by the server when it starts (app.py).
				handleHttpError: ({ path, message }) => {
					if (['/manifest.webmanifest', '/apple-touch-icon.png'].includes(path)) return;
					throw new Error(message);
				}
			}
		})
	]
});
