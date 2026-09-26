<script lang="ts">
	// The games already played, newest first, as a grid of videos to watch again: a frame
	// from the war, its length, how it ended, the map and side, the PC and when. Those
	// kept as they looked live play at 30 frames a second; the rest from the recording.
	import { get } from './api';
	import { SIDE, VERDICT, ago, arena, clock } from './format';
	import type { Played } from './types';

	let { onopen }: { onopen: (game: Played) => void } = $props();

	const PAGE = 24;
	let limit = $state(PAGE);
	let played: Played[] = $state([]);
	let loaded = $state(false);

	async function load() {
		const games = await get<Played[]>(`api/games?limit=${limit}`);
		if (games) played = games;
		loaded = true;
	}
	$effect(() => {
		void limit;
		load();
		const timer = setInterval(load, 60000);
		return () => clearInterval(timer);
	});
</script>

<ul>
	{#each played as g (g.game)}
		<li>
			<button onclick={() => onopen(g)}>
				<div class="thumb">
					<img
						src={`thumbs/${g.game}.jpg`}
						alt=""
						loading="lazy"
						onerror={(e) => ((e.currentTarget as HTMLImageElement).style.visibility = 'hidden')}
					/>
					<span class="result {g.result}">{VERDICT[g.result]}</span>
					<span class="time">{clock(g.seconds)}</span>
					{#if g.archived}<span class="fps">30 fps</span>{/if}
				</div>
				<div class="title">
					{arena(g.arena)} <span class={g.side}>{SIDE[g.side] ?? g.side}</span>
				</div>
				<div class="muted">
					{[g.label ?? g.station, g.plan ? `${g.plan} plan` : '', ago(g.ended_unix ?? g.started_unix)]
						.filter(Boolean)
						.join(' · ')}
				</div>
			</button>
		</li>
	{:else}
		{#if loaded}<li class="muted">No games in the last 30 days.</li>{/if}
	{/each}
</ul>
{#if played.length >= limit}
	<div class="more"><button class="chip" onclick={() => (limit += PAGE)}>Show more</button></div>
{/if}

<style>
	ul {
		list-style: none;
		margin: 0;
		padding: 0 16px;
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
		gap: 14px;
	}
	li > button {
		display: block;
		width: 100%;
		padding: 0;
		background: none;
		border: 0;
		text-align: left;
		cursor: pointer;
	}
	.thumb {
		position: relative;
		aspect-ratio: 16 / 9;
		background: var(--panel);
		border-radius: 10px;
		overflow: hidden;
	}
	img {
		width: 100%;
		height: 100%;
		object-fit: cover;
		display: block;
	}
	.result,
	.time,
	.fps {
		position: absolute;
		font-size: 12px;
		font-weight: 700;
		padding: 1px 7px;
		border-radius: 5px;
		background: rgba(0, 0, 0, 0.78);
	}
	.result {
		top: 8px;
		left: 8px;
		text-transform: capitalize;
	}
	.time {
		bottom: 8px;
		right: 8px;
		font-variant-numeric: tabular-nums;
	}
	.fps {
		bottom: 8px;
		left: 8px;
		color: var(--muted);
		font-weight: 600;
	}
	.title {
		margin-top: 6px;
		font-weight: 650;
	}
	.muted {
		font-size: 13px;
	}
	.more {
		display: flex;
		justify-content: center;
		padding: 16px;
	}
</style>
