<script lang="ts">
	// A PC's live stream (HLS). Attached while `src` is set and gone the moment it is not,
	// so a finished game's last seconds never pass for live ones; started again at the live
	// edge when the picture freezes for 20 s while it should play.
	import { onDestroy } from 'svelte';
	import type Hls from 'hls.js';
	import { nativeHls } from './format';

	let { src }: { src: string | null } = $props();

	let video: HTMLVideoElement;
	let hls: Hls | null = null;
	let lastTime = -1;
	let still = 0;

	function detach() {
		hls?.destroy();
		hls = null;
		if (video) {
			video.removeAttribute('src');
			video.load();
		}
	}

	async function attach(source: string) {
		detach();
		still = 0;
		const url = `${source}?${Date.now()}`;
		if (nativeHls()) {
			video.src = url;
		} else {
			const { default: HlsPlayer } = await import('hls.js');
			if (!HlsPlayer.isSupported()) return;
			hls = new HlsPlayer({ liveSyncDurationCount: 2 });
			// A playlist not written yet (a stream just starting) is fatal to hls.js: again soon.
			hls.on(HlsPlayer.Events.ERROR, (_, data) => {
				if (data.fatal) setTimeout(() => src && attach(src), 3000);
			});
			hls.loadSource(url);
			hls.attachMedia(video);
		}
		video.play().catch(() => {});
	}

	$effect(() => {
		if (src) attach(src);
		else detach();
	});

	const watchdog = setInterval(() => {
		if (!src || !video || video.paused) return;
		if (video.currentTime === lastTime) {
			if (++still >= 4) attach(src);
		} else still = 0;
		lastTime = video.currentTime;
	}, 5000);

	// Back in the app after a while away: straight to the live edge.
	function visible() {
		if (!document.hidden && src) attach(src);
	}

	onDestroy(() => {
		clearInterval(watchdog);
		detach();
	});
</script>

<svelte:document onvisibilitychange={visible} />

<!-- svelte-ignore a11y_media_has_caption -->
<video
	bind:this={video}
	autoplay
	muted
	playsinline
	controls
	onerror={() => src && setTimeout(() => src && attach(src), 3000)}
></video>

<style>
	video {
		display: block;
		width: 100%;
		aspect-ratio: 16 / 9;
		max-height: 72vh;
		background: #000;
	}
</style>
