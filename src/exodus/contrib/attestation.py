"""Contribution attestations.

A node that performed work for the network emits a :class:`ContributionClaim`
describing what it did (model, tokens served, time spent).  The claim is then
signed by the node into a :class:`SignedContribution` that any other node can
verify.  Claims are what the consensus layer orders and records in the ledger.
"""

from __future__ import annotations

from enum import Enum

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import BaseModel, ConfigDict, Field

from exodus.crypto import canonical_bytes, node_id_from_public_key, sign, verify


class WorkType(str, Enum):
    """The kind of work a node performed for the network."""

    text_generation = "text_generation"
    embedding = "embedding"
    image_generation = "image_generation"
    audio_generation = "audio_generation"
    generic = "generic"


class Precision(str, Enum):
    """Model precision and the relative compute cost factor."""

    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"
    fp8 = "fp8"
    int8 = "int8"
    int4 = "int4"
    int2 = "int2"


PRECISION_FACTORS: dict[Precision, float] = {
    Precision.fp32: 2.0,
    Precision.fp16: 1.0,
    Precision.bf16: 1.0,
    Precision.fp8: 0.6,
    Precision.int8: 0.6,
    Precision.int4: 0.35,
    Precision.int2: 0.2,
}


class DeviceTier(str, Enum):
    """Broad class of the accelerator that performed the work.

    Used to weight wall-clock time when cross-checking claimed FLOPS: the same
    token stream is expected to take far less wall time on a data-centre GPU
    than on a phone CPU.
    """

    cpu = "cpu"
    gpu_apple = "gpu_apple"
    gpu_nvidia = "gpu_nvidia"
    gpu_amd = "gpu_amd"
    tpu = "tpu"
    other = "other"


class ContributionClaim(BaseModel):
    """Self-reported description of a unit of work performed for the network.

    Everything here is *attested* by the signing node; honesty is enforced
    downstream by FLOPS sanity checks, double-claim detection and the quorum.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    claim_id: str = Field(description="Globally unique claim identifier (uuid).")
    node_id: str = Field(description="Signer node id (derived from its key).")
    seq: int = Field(ge=0, description="Monotonically increasing per-node sequence.")
    work_type: WorkType
    model_id: str = Field(description="e.g. 'mlx-community/Llama-3.2-1B-Instruct-4bit'.")
    params_b: float = Field(gt=0.0, description="Model parameter count, in billions.")
    precision: Precision
    prompt_tokens: int = Field(ge=0, description="Prompt tokens processed (prefill).")
    completion_tokens: int = Field(ge=0, description="Completion tokens generated.")
    compute_seconds: float = Field(
        ge=0.0, description="Wall-clock time spent on the accelerator."
    )
    flops_estimate: float = Field(
        gt=0.0, description="Node's own estimate of FLOPs performed."
    )
    device_tier: DeviceTier = DeviceTier.gpu_apple
    started_at: str = Field(description="ISO-8601 UTC start timestamp.")
    ended_at: str = Field(description="ISO-8601 UTC end timestamp.")
    last_seen_checkpoint_height: int = Field(
        ge=-1, description="Chain height this node had seen when claiming."
    )
    last_seen_checkpoint_hash: str = Field(
        description="Chain head hash this node had seen when claiming."
    )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def payload_bytes(self) -> bytes:
        """Canonical bytes for signing/hashing (excludes nothing; the model is the payload)."""

        return canonical_bytes(self.model_dump(mode="json"))

    def canonical(self) -> dict:
        return self.model_dump(mode="json")


class SignedContribution(BaseModel):
    """A contribution claim plus the signature that makes it attributable."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    claim: ContributionClaim
    public_key_hex: str = Field(description="Hex-encoded Ed25519 public key.")
    signature_hex: str = Field(description="Hex-encoded Ed25519 signature.")

    def verify(self) -> bool:
        """Return ``True`` when the signature matches the claim and the node id."""

        try:
            public_key = bytes.fromhex(self.public_key_hex)
        except ValueError:
            return False
        try:
            signature = bytes.fromhex(self.signature_hex)
        except ValueError:
            return False

        if node_id_from_public_key(public_key) != self.claim.node_id:
            return False
        return verify(self.claim.payload_bytes(), signature, public_key)

    def bytes_for_protocol(self) -> bytes:
        return canonical_bytes(self.model_dump(mode="json"))

    @classmethod
    def create(cls, claim: ContributionClaim, private_key: bytes) -> SignedContribution:
        """Sign a claim, deriving the public key and node id from *private_key*."""

        key = Ed25519PrivateKey.from_private_bytes(private_key)
        public_key = key.public_key().public_bytes_raw()
        signature = sign(claim.payload_bytes(), private_key)
        return cls(
            claim=claim,
            public_key_hex=public_key.hex(),
            signature_hex=signature.hex(),
        )
