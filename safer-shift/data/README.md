# SaferShift Data Storage

## Storage Organization
- `profiles/`: Local JSON driver profiles containing derived threshold statistics (open-eye EAR distributions, calibrated neutral head pose).
- `logs/`: Local telemetry CSV logs recording periodic samples and immediate safety alert triggers.

## Privacy & Zero-Retention Policy
- **No Raw Video Footage**: SaferShift NEVER records or stores raw webcam or video camera footage to disk.
- **No Facial Recognition**: Profiles do not contain embeddings, image hashes, or identity verification keys. They only contain numerical threshold statistics for safety tuning.
- **Profile Deletion**: Profiles can be permanently deleted via the Streamlit dashboard or CLI (`ProfileManager.delete_profile(id)`).
