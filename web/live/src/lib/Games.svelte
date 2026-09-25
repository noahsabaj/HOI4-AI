<script lang="ts">
	// The games already played, newest first; one tapped opens its replay.
	import { get } from './api';
	import { SIDE, VERDICT, arena, clock, day } from './format';
	import type { Played } from './types';

	let { onopen }: { onopen: (game: Played) => void } = $props();
	let played: Played[] = $state([]);

	get<Played[]>('api/games?limit=80').then((games) => (played = games ?? []));
</script>

<ol>
	{#each played as g (g.game)}
		<li>
			<button onclick={() => onopen(g)}>
				<span
					><time>{day(g.started_unix)}</time>{arena(g.arena)}
					<span class={g.side}>{SIDE[g.side] ?? g.side}</span>
					<span class="muted">{g.plan ?? ''}</span></span
				>
				<span class={g.result}>{VERDICT[g.result]} {clock(g.seconds)} ▶</span>
			</button>
		</li>
	{:else}
		<li class="muted">No games in the last 30 days.</li>
	{/each}
</ol>

<style>
	ol {
		list-style: none;
		margin: 0;
		padding: 0;
		font-size: 14px;
	}
	button {
		display: flex;
		justify-content: space-between;
		gap: 8px;
		width: 100%;
		padding: 8px 0;
		background: none;
		border: 0;
		border-bottom: 1px solid var(--line);
		text-align: left;
		cursor: pointer;
	}
</style>
