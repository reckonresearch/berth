# Running the holdout at your gateway

The assignment must not require application code. Shipping a vendor's function
into a production request path, through review and deploy, for a company still
being evaluated, is where most integrations die.

Every gateway below does weighted routing with consistent hashing natively.
The configuration change is small, nothing of ours runs in the request path,
and you can audit the assignment without reading our code.

## What you need

Two endpoints, baseline and treatment, and the seed from the declaration. The
seed is published before the period opens; that is the defence against
selective measurement, so do not generate your own.

## Envoy

```yaml
route:
  weighted_clusters:
    clusters:
      - name: baseline
        weight: 5          # the declared holdout fraction, as a percentage
      - name: treatment
        weight: 95
    total_weight: 100
  hash_policy:
    - header:
        header_name: x-request-id
      terminal: true
```

Envoy hashes the header you name. Set `x-request-id` to a UUID upstream if it
is not one already: sequential or tenant-prefixed identifiers carry structure
that can survive a weak hash and land one tenant disproportionately on one leg.

## NGINX

```nginx
split_clients "${seed}${request_id}" $placement_leg {
    5%      baseline;
    *       treatment;
}

upstream baseline  { server baseline.internal:8000; }
upstream treatment { server treatment.internal:8000; }

server {
    location / {
        proxy_pass http://$placement_leg;
    }
}
```

`split_clients` uses MurmurHash2 over the concatenation, which is why the seed
goes first: it removes leading structure from the identifier.

## AWS Application Load Balancer

Two target groups with a weighted forward action, 5 and 95, and
`stickiness.enabled = false`. Turn stickiness off explicitly. Session affinity
assigns per client rather than per request, which is exactly what the protocol
forbids: a coarser assignment lets a per-customer pattern land on one leg and
the difference becomes a property of the traffic.

## Istio

```yaml
http:
  - route:
      - destination: { host: baseline }
        weight: 5
      - destination: { host: treatment }
        weight: 95
```

## If you have no gateway

Use the reference implementation directly. It is about thirty lines and it is
in `berth/holdout.py`:

```python
from berth.holdout import assign

leg = assign(request_id, seed="<from the declaration>", holdout_fraction=0.05)
```

Call it **after** the response is produced, never before. If a request can be
identified as holdout on the way in, it can be treated differently, and the
comparison then measures the treatment rather than the placement.

## Before the period opens

Check that the hash distributes over your identifier format:

```
berth holdout check --seed <seed> --fraction 0.05 --ids sample_ids.txt
```

A realised fraction outside one percentage point of the declared one is a
defect in the instrument rather than a result, and it is much cheaper to find
here than in settlement.
