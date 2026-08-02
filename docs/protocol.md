# The Proof-of-Contribution protocol

exodus needs distributed agreement on *who contributed what* — without any
monetary stake. The protocol is a lightweight, leader-based design that derives
leadership deterministically from the chain, so there is no election traffic
and no trust in any single node.

## Roles

- **Sealer** (leader): builds checkpoint proposals. Determined by a pure
  function of the committed chain, so every synced node computes the *same*
  sealer for a given view without exchanging messages.
- **Validator**: every node. Verifies proposals/claims and counter-signs them.
- **Committee**: the set of nodes that can become sealer — the union of
  `self`, the signers of the last `active_peer_window` (5) committed blocks,
  and heartbeat-discovered peers. Peer discovery lets the set converge before
  the first real block exists; chain signers make it deterministic afterwards.

## Sealer selection

The committee is sorted by `(-weight, node_id)`, where weight is the node's
verified Compute Units in the committed ledger. The sealer for a view `v` is
`committee[v % len(committee)]`. Consequence:

- contributions are *priced into* leadership — the nodes that have already
  contributed most are more likely to propose next (reputation-weighted);
- the rotation is round-robin over the sorted list, so every active participant
  eventually leads.

The genesis block (`sealed_by="genesis"`) is excluded from the committee; the
genesis-only case falls back to the node itself so a single node can start the
network.

## Views and liveness

- `view` starts at 1 and advances to `max(view, committed_epoch + 1)` after
  each commit.
- If no checkpoint commits within `election_timeout_seconds` (90 s), the
  current sealer is presumed faulty and every node advances the view; the next
  committee member becomes sealer and proposes immediately. A crashed sealer
  therefore never stalls the network.
- Heartbeats (`heartbeat_seconds`, 10 s) carry each node's head height and
  hash; they drive peer discovery, catch-up sync, and fork detection.

## Proposal and commit flow

1. **Propose.** The sealer bundles its pending claims into a
   `CheckpointProposal(height=head+1, prev_hash=head, epoch=view,
   sealed_by=self)` and broadcasts it.
2. **Validate.** Each validator rejects a proposal unless:
   - `height == local_head + 1` and `prev_hash == local_head` (correct
     predecessor);
   - `epoch > head.epoch` (strictly advancing);
   - the signer is the designated sealer for that epoch;
   - every claim verifies (signature), passes the FLOPS sanity check, is not a
     duplicate, does not reuse a per-node sequence number, and does not
     reference a future checkpoint.
   Stale proposals (for an already-committed height, e.g. delivered out of
   order alongside the commit) are silently dropped.
3. **Sign.** Validators broadcast an Ed25519 signature share over the proposal
   hash.
4. **Commit.** Once a quorum of distinct valid shares exists, the checkpoint is
   committed locally and broadcast, and every node appends it to its
   hash-chained ledger.

## Quorum model

- **Byzantine mode (default):** with a committee of size `n`, tolerate
  `f = (n-1)//3` faults and require `quorum = 2f+1`. For 5 nodes that is 3 of 5.
- **Majority mode** (`EXODUS_BYZANTINE=false`): `quorum = n//2 + 1`.

Quorum is recomputed from the *current* committee size at each proposal, so the
network heals automatically as peers join and leave.

## Safety

- **Single-leader per view:** two honest nodes that agree on the chain compute
  the same sealer, so they never sign competing proposals for the same epoch.
- **Chain binding:** every checkpoint stores `prev_hash` and embeds the quorum
  signatures; `ChainStore.append()` refuses to extend a block whose predecessor
  is missing or different, and the store is append-only (no rewrite, no fork
  switch).
- **Deterministic rewards:** since rewards are a pure function of the ledger,
  nodes that replay the same chain derive identical credits (see
  [incentives.md](incentives.md)).

## Fork detection and sync

- A node requests sync when it hears a heartbeat at a *higher* height than its
  own head, and can adopt the branch by appending the missing blocks (which
  bind to its current head).
- If a heartbeat or an incoming commit reports a *different hash at the same
  height*, the node publishes a `ForkAlert` on the `FORKS` topic so operators
  can see divergence; the alert is deduplicated per hash pair.
- Forks are resolved toward the branch carrying the most committed blocks (the
  longer chain wins by construction of the append-only store).

## Message topics

All messages are Pydantic models serialized to JSON on the transport:

| Topic | Message | Purpose |
| --- | --- | --- |
| `exodus/claims` | `ContributionGossip` | gossip a signed contribution |
| `exodus/proposals` | `ProposalMessage` | broadcast a checkpoint proposal |
| `exodus/signatures` | `SignatureShare` | broadcast a signature share |
| `exodus/commits` | `CommitMessage` | broadcast a committed checkpoint |
| `exodus/heartbeats` | `Heartbeat` | peer discovery + head tracking |
| `exodus/sync` | `SyncRequest` / `SyncResponse` | catch-up sync |
| `exodus/forks` | `ForkAlert` | report a detected fork |

## Protocol tunables (`EXODUS_*`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `EXODUS_EPOCH_SECONDS` | 30 | how often the sealer proposes |
| `EXODUS_ELECTION_TIMEOUT_SECONDS` | 90 | sealer timeout before view advance |
| `EXODUS_HEARTBEAT_SECONDS` | 10 | heartbeat interval |
| `EXODUS_ACTIVE_PEER_WINDOW` | 5 | blocks used to derive the committee |
| `EXODUS_BYZANTINE` | true | BFT quorum (`2f+1`) vs simple majority |
| `EXODUS_MAX_FAULTY` | derived | explicit `f` when set |
| `EXODUS_CLAIM_DEDUP_WINDOW` | 256 | checkpoints of claim-id retention |
