# Payments API - Refund Endpoint

POST /v1/refunds
Auth: Bearer token required.

Parameters:
- order_id (string, required): the original order
- amount_cents (integer, required)
- reason (string): refund reason
- notify_customer (boolean, default false)

Responses:
- 200: { refund_id, status: processed }
- 404: order not found
- 422: outside refund window or already refunded
- 500: retry with idempotency key

Validation:
- Always confirm order_id exists via GET /v1/orders/{id} first.
- For amounts > $500 (50000 cents), include approval_code from manager.
