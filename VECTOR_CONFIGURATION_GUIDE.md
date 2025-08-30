# ClaudeWatch Vector Configuration Guide

ClaudeWatch now supports **three different methods** for configuring behavioral detection vectors:

## 1. 🔄 Auto-Generated Vectors (Default)

Uses example datasets to automatically generate discriminative vectors via `contrast()`.

```json
{
  "good_examples_path": "/path/to/good_examples.json",
  "bad_examples_path": "/path/to/bad_examples.json", 
  "model": "meta-llama/Llama-3.3-70B-Instruct",
  "alert_strategy": "ratio"
}
```

**Pros:** Automatic, requires only example datasets  
**Cons:** May capture linguistic artifacts rather than behavioral patterns

## 2. 📁 Custom Vector Files

Points to pre-generated vector files using `_vector_source`.

```json
{
  "good_examples_path": ["/path/to/examples1.json", "/path/to/examples2.json"],
  "bad_examples_path": "/path/to/bad_examples.json",
  "model": "meta-llama/Llama-3.3-70B-Instruct", 
  "_vector_source": "curated_coaching_vectors_Llama_3.3_70B_Instruct.json",
  "alert_strategy": "ratio"
}
```

**Pros:** Uses hand-curated, behaviorally-specific vectors  
**Cons:** Requires separate vector file creation step

## 3. 🎯 Direct Vector Specification (New!)

Directly specifies vector UUIDs in the configuration file.

```json
{
  "direct_vectors": {
    "good": [
      {
        "uuid": "cbed4926-5684-4090-a443-e2c290d5d8be",
        "label": "Active listening and empathetic presence in counseling contexts"
      },
      {
        "uuid": "6d255487-97a3-48e2-a222-5a9adc1415f2", 
        "label": "Empathetic acknowledgment of emotional pain or difficulty"
      }
    ],
    "bad": [
      {
        "uuid": "880cc089-19c3-40ca-9dc1-103eb3a4f5ee",
        "label": "The assistant is telling the user what they can or should do"
      },
      {
        "uuid": "e3e49f49-c963-43ef-b1e6-a4825fc82079",
        "label": "Dismissive responses minimizing relationship concerns"
      }
    ]
  },
  "model": "meta-llama/Llama-3.3-70B-Instruct",
  "alert_strategy": "ratio"
}
```

**Pros:** 
- No external files needed
- Precise control over vectors
- Easy sharing and experimentation  
- Perfect for A/B testing different vector combinations

**Cons:** Requires knowing specific vector UUIDs

## Vector Specification Format

For direct vectors, each vector object supports:

- **`uuid`** (required): Goodfire vector UUID
- **`label`** (optional): Human-readable description
- **`index_in_sae`** (optional): SAE index (auto-populated if needed)

## Finding Vector UUIDs

Use the Goodfire API to search for relevant vectors:

```python
from goodfire import Client
client = Client(api_key="your_key")

# Search for specific behavioral patterns
results = client.features.search("empathetic listening", 
                                 model="meta-llama/Llama-3.3-70B-Instruct", 
                                 top_k=5)
for feature in results:
    print(f"UUID: {feature.uuid}")
    print(f"Label: {feature.label}")
```

## Configuration Priority

ClaudeWatch loads vectors in this priority order:

1. **Direct vectors** (`direct_vectors` field)
2. **Custom vector file** (`_vector_source` field)  
3. **Auto-generated** (from example paths)

## Examples

See these example configurations:
- `configs/direct_vectors_example.json` - Direct vector specification
- `configs/curated_coaching_vectors.json` - Custom vector file
- `configs/expanded_diverse_coaching.json` - Auto-generated vectors

## Validation

The system validates that you provide either:
- `good_examples_path` AND `bad_examples_path` (for auto-generated)
- OR `direct_vectors` with 'good' and 'bad' lists
- OR `_vector_source` pointing to existing vector file

## Use Cases

- **Direct vectors**: Quick experiments, sharing specific configurations
- **Custom vector files**: Production systems with curated, tested vectors
- **Auto-generated**: Rapid prototyping with new datasets

Choose the method that best fits your workflow and requirements!