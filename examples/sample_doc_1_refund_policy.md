# Customer Refund Procedure

## Purpose
Handle customer refund requests within 5 business days.

## Steps
1. Collect the order ID, customer email, and reason for refund from the support ticket.
2. Verify the purchase in the Orders API and check it is within the 30-day refund window.
3. If amount is over $500, request manager approval via email before proceeding.
4. Process the refund via the Payments API and record the transaction ID.
5. Notify the customer by email with the refund confirmation and expected timeline.
6. Log the case in the CRM and close the ticket.

## Error handling
- If order not found, ask customer for proof of purchase.
- If payment API fails, retry once then escalate to finance.
