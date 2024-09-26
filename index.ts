import express from 'express';
import { middleware, WebhookEvent, TextMessage, MessageEvent, Client, validateSignature, Profile } from '@line/bot-sdk';
import dotenv from 'dotenv';
import fetch from 'node-fetch';

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

const config = {
    channelAccessToken: process.env.LINE_CHANNEL_ACCESS_TOKEN!,
    channelSecret: process.env.LINE_CHANNEL_SECRET!
};

const client = new Client(config);

// Simple in-memory queue implementation
class SimpleQueue {
    private queue: any[] = [];
    private processing: boolean = false;

    add(item: any) {
        this.queue.push(item);
        if (!this.processing) {
            this.process();
        }
    }

    private async process() {
        this.processing = true;
        while (this.queue.length > 0) {
            const item = this.queue.shift();
            try {
                await this.processItem(item);
            } catch (error) {
                console.error('Error processing item:', error);
            }
        }
        this.processing = false;
    }

    private async processItem(item: any) {
        const { userId, message } = item;
        console.log(`userId: ${userId}`);
        console.log(`Processing message: ${message}`);

        try {
            const llmResponse = await CallLLM(userId, message);
            await sendMessage(userId, `~ ${llmResponse}`);
        } catch (error) {
            console.error('Error calling LLM:', error);
            await sendMessage(userId, "Sorry, I couldn't process your message at this time.");
        }
    }
}

const messageQueue = new SimpleQueue();

app.use(express.json());

app.post('/test', (req, res, next) => {
    console.log('Received test request');
    console.log('Headers:', req.headers);
    console.log('Body:', req.body);
    console.log('config', config);
    // res.sendStatus(200);

    console.log(validateSignature(req.body, config.channelSecret, req.headers['x-line-signature'] as string));

    res.sendStatus(200);
});
// , middleware(config), (req, res) => {
//   res.sendStatus(200);
// });

// Updated LINE webhook route with message queue
app.post('/webhook', (req, res) => {
    console.log('Received webhook request');
    console.log('Headers:', req.headers);
    console.log('Body:', req.body);

    const events: WebhookEvent[] = req.body.events;

    // Add events to the queue
    events.forEach((event: WebhookEvent) => {
        if (event.type === 'message' && event.message.type === 'text') {
            messageQueue.add({
                userId: (event as MessageEvent).source.userId,
                message: (event.message as TextMessage).text
            });
        }
    });

    // Send immediate success response
    res.sendStatus(200);
});

// Private function for sending messages
async function sendMessage(userId: string, message: string): Promise<void> {
    if (!userId || !message) {
        throw new Error('userId and message are required');
    }

    try {
        await client.pushMessage(userId, {
            type: 'text',
            text: message
        });
    } catch (error) {
        console.error('Error sending message:', error);
        throw error;
    }
}

// New function to call the LLM API
async function CallLLM(userId: string, input: string): Promise<string> {
    let displayName = "User";

    try {
        // Fetch the user's profile
        const profile: Profile = await client.getProfile(userId);
        displayName = profile.displayName;
    } catch (error) {
        console.error('Error fetching user profile:', error);
    }

    const url = process.env.LLM_API_URL || '';
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'x-api-key': 'sk-1AwVFcwwANKVQL-Ll7vm5eKr5sHnXuvO_Lxinr2bBuA'
        },
        body: JSON.stringify({
            "output_type": "chat",
            "input_type": "chat",
            "tweaks": {
                "ChatInput-FHzEA": {
                    "files": "",
                    "input_value": input,
                    "sender": "User",
                    "sender_name": "",
                    "session_id": "",
                    "should_store_message": true
                },
                "AstraVectorStoreComponent-vsEJl": {
                    "api_endpoint": "https://8bb671cd-5109-425e-9dbf-c9cbe6d3dc8e-us-east1.apps.astra.datastax.com",
                    "batch_size": null,
                    "bulk_delete_concurrency": null,
                    "bulk_insert_batch_concurrency": null,
                    "bulk_insert_overwrite_concurrency": null,
                    "collection_indexing_policy": "",
                    "collection_name": "bearattacks2",
                    "metadata_indexing_exclude": "",
                    "metadata_indexing_include": "",
                    "metric": "",
                    "namespace": "",
                    "number_of_results": 4,
                    "pre_delete_collection": false,
                    "search_filter": {},
                    "search_input": "",
                    "search_score_threshold": 0,
                    "search_type": "Similarity",
                    "setup_mode": "Sync",
                    "token": "AstraCS:xOagPjkIZbvewNBsSbtNGUMa:9b6128514d5e2eaa16e04da2e77d9a88b3e4e738eae043c3022d0197af278ac8"
                },
                "ParseData-LQ6Cp": {
                    "sep": "\n",
                    "template": "{text}"
                },
                "Prompt-jLeoA": {
                    "context": "",
                    "question": "",
                    "template": "{context}\n\n---\nYour name is Sam.\nRespond to the user with humor, kindness, and emojis.\nGiven the context above, answer the question as best as possible.\nAnswer in the same language as the question.\n\nChat History:\n{history}\n\n\nQuestion: {question}\nAnswer: ",
                    "history": ""
                },
                "ChatOutput-QGdCY": {
                    "data_template": "{text}",
                    "input_value": "",
                    "sender": "Machine",
                    "sender_name": "AI",
                    "session_id": "",
                    "should_store_message": true
                },
                "OpenAIEmbeddings-SWc6a": {
                    "chunk_size": 1000,
                    "client": "",
                    "default_headers": {},
                    "default_query": {},
                    "deployment": "",
                    "dimensions": null,
                    "embedding_ctx_length": 1536,
                    "max_retries": 3,
                    "model": "text-embedding-ada-002",
                    "model_kwargs": {},
                    "openai_api_base": "",
                    "openai_api_key": "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",
                    "openai_api_type": "",
                    "openai_api_version": "",
                    "openai_organization": "",
                    "openai_proxy": "",
                    "request_timeout": null,
                    "show_progress_bar": false,
                    "skip_empty": false,
                    "tiktoken_enable": true,
                    "tiktoken_model_name": ""
                },
                "OpenAIModel-t5IYV": {
                    "api_key": "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",
                    "input_value": "",
                    "json_mode": false,
                    "max_tokens": null,
                    "model_kwargs": {},
                    "model_name": "gpt-4o",
                    "openai_api_base": "",
                    "output_schema": {},
                    "seed": 1,
                    "stream": false,
                    "system_message": "",
                    "temperature": 0.6
                },
                "groupComponent-ZSgEm": {
                    "chunk_overlap_SplitText-fzMrJ": 200,
                    "chunk_size_SplitText-fzMrJ": 1000,
                    "code_SplitText-fzMrJ": "from typing import List\n\nfrom langchain_text_splitters import CharacterTextSplitter\n\nfrom langflow.custom import Component\nfrom langflow.io import HandleInput, IntInput, MessageTextInput, Output\nfrom langflow.schema import Data\nfrom langflow.utils.util import unescape_string\n\n\nclass SplitTextComponent(Component):\n    display_name: str = \"Split Text\"\n    description: str = \"Split text into chunks based on specified criteria.\"\n    icon = \"scissors-line-dashed\"\n    name = \"SplitText\"\n\n    inputs = [\n        HandleInput(\n            name=\"data_inputs\",\n            display_name=\"Data Inputs\",\n            info=\"The data to split.\",\n            input_types=[\"Data\"],\n            is_list=True,\n        ),\n        IntInput(\n            name=\"chunk_overlap\",\n            display_name=\"Chunk Overlap\",\n            info=\"Number of characters to overlap between chunks.\",\n            value=200,\n        ),\n        IntInput(\n            name=\"chunk_size\",\n            display_name=\"Chunk Size\",\n            info=\"The maximum number of characters in each chunk.\",\n            value=1000,\n        ),\n        MessageTextInput(\n            name=\"separator\",\n            display_name=\"Separator\",\n            info=\"The character to split on. Defaults to newline.\",\n            value=\"\\n\",\n        ),\n    ]\n\n    outputs = [\n        Output(display_name=\"Chunks\", name=\"chunks\", method=\"split_text\"),\n    ]\n\n    def _docs_to_data(self, docs):\n        data = []\n        for doc in docs:\n            data.append(Data(text=doc.page_content, data=doc.metadata))\n        return data\n\n    def split_text(self) -> List[Data]:\n        separator = unescape_string(self.separator)\n\n        documents = []\n        for _input in self.data_inputs:\n            if isinstance(_input, Data):\n                documents.append(_input.to_lc_document())\n\n        splitter = CharacterTextSplitter(\n            chunk_overlap=self.chunk_overlap,\n            chunk_size=self.chunk_size,\n            separator=separator,\n        )\n        docs = splitter.split_documents(documents)\n        data = self._docs_to_data(docs)\n        self.status = data\n        return data\n",
                    "separator_SplitText-fzMrJ": "\n",
                    "code_File-SUAe4": "from pathlib import Path\n\nfrom langflow.base.data.utils import TEXT_FILE_TYPES, parse_text_file_to_data\nfrom langflow.custom import Component\nfrom langflow.io import BoolInput, FileInput, Output\nfrom langflow.schema import Data\n\n\nclass FileComponent(Component):\n    display_name = \"File\"\n    description = \"A generic file loader.\"\n    icon = \"file-text\"\n    name = \"File\"\n\n    inputs = [\n        FileInput(\n            name=\"path\",\n            display_name=\"Path\",\n            file_types=TEXT_FILE_TYPES,\n            info=f\"Supported file types: {', '.join(TEXT_FILE_TYPES)}\",\n        ),\n        BoolInput(\n            name=\"silent_errors\",\n            display_name=\"Silent Errors\",\n            advanced=True,\n            info=\"If true, errors will not raise an exception.\",\n        ),\n    ]\n\n    outputs = [\n        Output(display_name=\"Data\", name=\"data\", method=\"load_file\"),\n    ]\n\n    def load_file(self) -> Data:\n        if not self.path:\n            raise ValueError(\"Please, upload a file to use this component.\")\n        resolved_path = self.resolve_path(self.path)\n        silent_errors = self.silent_errors\n\n        extension = Path(resolved_path).suffix[1:].lower()\n\n        if extension == \"doc\":\n            raise ValueError(\"doc files are not supported. Please save as .docx\")\n        if extension not in TEXT_FILE_TYPES:\n            raise ValueError(f\"Unsupported file type: {extension}\")\n\n        data = parse_text_file_to_data(resolved_path, silent_errors)\n        self.status = data if data else \"No data\"\n        return data or Data()\n",
                    "path_File-SUAe4": "",
                    "silent_errors_File-SUAe4": false,
                    "api_endpoint_AstraVectorStoreComponent-86DpN": "",
                    "batch_size_AstraVectorStoreComponent-86DpN": null,
                    "bulk_delete_concurrency_AstraVectorStoreComponent-86DpN": null,
                    "bulk_insert_batch_concurrency_AstraVectorStoreComponent-86DpN": null,
                    "bulk_insert_overwrite_concurrency_AstraVectorStoreComponent-86DpN": null,
                    "code_AstraVectorStoreComponent-86DpN": "from loguru import logger\n\nfrom langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store\nfrom langflow.helpers import docs_to_data\nfrom langflow.inputs import DictInput, FloatInput\nfrom langflow.io import (\n    BoolInput,\n    DataInput,\n    DropdownInput,\n    HandleInput,\n    IntInput,\n    MultilineInput,\n    SecretStrInput,\n    StrInput,\n)\nfrom langflow.schema import Data\n\n\nclass AstraVectorStoreComponent(LCVectorStoreComponent):\n    display_name: str = \"Astra DB\"\n    description: str = \"Implementation of Vector Store using Astra DB with search capabilities\"\n    documentation: str = \"https://python.langchain.com/docs/integrations/vectorstores/astradb\"\n    name = \"AstraDB\"\n    icon: str = \"AstraDB\"\n\n    inputs = [\n        StrInput(\n            name=\"collection_name\",\n            display_name=\"Collection Name\",\n            info=\"The name of the collection within Astra DB where the vectors will be stored.\",\n            required=True,\n        ),\n        SecretStrInput(\n            name=\"token\",\n            display_name=\"Astra DB Application Token\",\n            info=\"Authentication token for accessing Astra DB.\",\n            value=\"ASTRA_DB_APPLICATION_TOKEN\",\n            required=True,\n        ),\n        SecretStrInput(\n            name=\"api_endpoint\",\n            display_name=\"API Endpoint\",\n            info=\"API endpoint URL for the Astra DB service.\",\n            value=\"ASTRA_DB_API_ENDPOINT\",\n            required=True,\n        ),\n        MultilineInput(\n            name=\"search_input\",\n            display_name=\"Search Input\",\n        ),\n        DataInput(\n            name=\"ingest_data\",\n            display_name=\"Ingest Data\",\n            is_list=True,\n        ),\n        StrInput(\n            name=\"namespace\",\n            display_name=\"Namespace\",\n            info=\"Optional namespace within Astra DB to use for the collection.\",\n            advanced=True,\n        ),\n        DropdownInput(\n            name=\"metric\",\n            display_name=\"Metric\",\n            info=\"Optional distance metric for vector comparisons in the vector store.\",\n            options=[\"cosine\", \"dot_product\", \"euclidean\"],\n            advanced=True,\n        ),\n        IntInput(\n            name=\"batch_size\",\n            display_name=\"Batch Size\",\n            info=\"Optional number of data to process in a single batch.\",\n            advanced=True,\n        ),\n        IntInput(\n            name=\"bulk_insert_batch_concurrency\",\n            display_name=\"Bulk Insert Batch Concurrency\",\n            info=\"Optional concurrency level for bulk insert operations.\",\n            advanced=True,\n        ),\n        IntInput(\n            name=\"bulk_insert_overwrite_concurrency\",\n            display_name=\"Bulk Insert Overwrite Concurrency\",\n            info=\"Optional concurrency level for bulk insert operations that overwrite existing data.\",\n            advanced=True,\n        ),\n        IntInput(\n            name=\"bulk_delete_concurrency\",\n            display_name=\"Bulk Delete Concurrency\",\n            info=\"Optional concurrency level for bulk delete operations.\",\n            advanced=True,\n        ),\n        DropdownInput(\n            name=\"setup_mode\",\n            display_name=\"Setup Mode\",\n            info=\"Configuration mode for setting up the vector store, with options like 'Sync', 'Async', or 'Off'.\",\n            options=[\"Sync\", \"Async\", \"Off\"],\n            advanced=True,\n            value=\"Sync\",\n        ),\n        BoolInput(\n            name=\"pre_delete_collection\",\n            display_name=\"Pre Delete Collection\",\n            info=\"Boolean flag to determine whether to delete the collection before creating a new one.\",\n            advanced=True,\n        ),\n        StrInput(\n            name=\"metadata_indexing_include\",\n            display_name=\"Metadata Indexing Include\",\n            info=\"Optional list of metadata fields to include in the indexing.\",\n            advanced=True,\n        ),\n        HandleInput(\n            name=\"embedding\",\n            display_name=\"Embedding or Astra Vectorize\",\n            input_types=[\"Embeddings\", \"dict\"],\n            info=\"Allows either an embedding model or an Astra Vectorize configuration.\",  # TODO: This should be optional, but need to refactor langchain-astradb first.\n        ),\n        StrInput(\n            name=\"metadata_indexing_exclude\",\n            display_name=\"Metadata Indexing Exclude\",\n            info=\"Optional list of metadata fields to exclude from the indexing.\",\n            advanced=True,\n        ),\n        StrInput(\n            name=\"collection_indexing_policy\",\n            display_name=\"Collection Indexing Policy\",\n            info=\"Optional dictionary defining the indexing policy for the collection.\",\n            advanced=True,\n        ),\n        IntInput(\n            name=\"number_of_results\",\n            display_name=\"Number of Results\",\n            info=\"Number of results to return.\",\n            advanced=True,\n            value=4,\n        ),\n        DropdownInput(\n            name=\"search_type\",\n            display_name=\"Search Type\",\n            info=\"Search type to use\",\n            options=[\"Similarity\", \"Similarity with score threshold\", \"MMR (Max Marginal Relevance)\"],\n            value=\"Similarity\",\n            advanced=True,\n        ),\n        FloatInput(\n            name=\"search_score_threshold\",\n            display_name=\"Search Score Threshold\",\n            info=\"Minimum similarity score threshold for search results. (when using 'Similarity with score threshold')\",\n            value=0,\n            advanced=True,\n        ),\n        DictInput(\n            name=\"search_filter\",\n            display_name=\"Search Metadata Filter\",\n            info=\"Optional dictionary of filters to apply to the search query.\",\n            advanced=True,\n            is_list=True,\n        ),\n    ]\n\n    @check_cached_vector_store\n    def build_vector_store(self):\n        try:\n            from langchain_astradb import AstraDBVectorStore\n            from langchain_astradb.utils.astradb import SetupMode\n        except ImportError:\n            raise ImportError(\n                \"Could not import langchain Astra DB integration package. \"\n                \"Please install it with `pip install langchain-astradb`.\"\n            )\n\n        try:\n            if not self.setup_mode:\n                self.setup_mode = self._inputs[\"setup_mode\"].options[0]\n\n            setup_mode_value = SetupMode[self.setup_mode.upper()]\n        except KeyError:\n            raise ValueError(f\"Invalid setup mode: {self.setup_mode}\")\n\n        if not isinstance(self.embedding, dict):\n            embedding_dict = {\"embedding\": self.embedding}\n        else:\n            from astrapy.info import CollectionVectorServiceOptions\n\n            dict_options = self.embedding.get(\"collection_vector_service_options\", {})\n            dict_options[\"authentication\"] = {\n                k: v for k, v in dict_options.get(\"authentication\", {}).items() if k and v\n            }\n            dict_options[\"parameters\"] = {k: v for k, v in dict_options.get(\"parameters\", {}).items() if k and v}\n            embedding_dict = {\n                \"collection_vector_service_options\": CollectionVectorServiceOptions.from_dict(dict_options)\n            }\n            collection_embedding_api_key = self.embedding.get(\"collection_embedding_api_key\")\n            if collection_embedding_api_key:\n                embedding_dict[\"collection_embedding_api_key\"] = collection_embedding_api_key\n\n        vector_store_kwargs = {\n            **embedding_dict,\n            \"collection_name\": self.collection_name,\n            \"token\": self.token,\n            \"api_endpoint\": self.api_endpoint,\n            \"namespace\": self.namespace or None,\n            \"metric\": self.metric or None,\n            \"batch_size\": self.batch_size or None,\n            \"bulk_insert_batch_concurrency\": self.bulk_insert_batch_concurrency or None,\n            \"bulk_insert_overwrite_concurrency\": self.bulk_insert_overwrite_concurrency or None,\n            \"bulk_delete_concurrency\": self.bulk_delete_concurrency or None,\n            \"setup_mode\": setup_mode_value,\n            \"pre_delete_collection\": self.pre_delete_collection or False,\n        }\n\n        if self.metadata_indexing_include:\n            vector_store_kwargs[\"metadata_indexing_include\"] = self.metadata_indexing_include\n        elif self.metadata_indexing_exclude:\n            vector_store_kwargs[\"metadata_indexing_exclude\"] = self.metadata_indexing_exclude\n        elif self.collection_indexing_policy:\n            vector_store_kwargs[\"collection_indexing_policy\"] = self.collection_indexing_policy\n\n        try:\n            vector_store = AstraDBVectorStore(**vector_store_kwargs)\n        except Exception as e:\n            raise ValueError(f\"Error initializing AstraDBVectorStore: {str(e)}\") from e\n\n        self._add_documents_to_vector_store(vector_store)\n        return vector_store\n\n    def _add_documents_to_vector_store(self, vector_store):\n        documents = []\n        for _input in self.ingest_data or []:\n            if isinstance(_input, Data):\n                documents.append(_input.to_lc_document())\n            else:\n                raise ValueError(\"Vector Store Inputs must be Data objects.\")\n\n        if documents:\n            logger.debug(f\"Adding {len(documents)} documents to the Vector Store.\")\n            try:\n                vector_store.add_documents(documents)\n            except Exception as e:\n                raise ValueError(f\"Error adding documents to AstraDBVectorStore: {str(e)}\") from e\n        else:\n            logger.debug(\"No documents to add to the Vector Store.\")\n\n    def _map_search_type(self):\n        if self.search_type == \"Similarity with score threshold\":\n            return \"similarity_score_threshold\"\n        elif self.search_type == \"MMR (Max Marginal Relevance)\":\n            return \"mmr\"\n        else:\n            return \"similarity\"\n\n    def _build_search_args(self):\n        args = {\n            \"k\": self.number_of_results,\n            \"score_threshold\": self.search_score_threshold,\n        }\n\n        if self.search_filter:\n            clean_filter = {k: v for k, v in self.search_filter.items() if k and v}\n            if len(clean_filter) > 0:\n                args[\"filter\"] = clean_filter\n        return args\n\n    def search_documents(self) -> list[Data]:\n        vector_store = self.build_vector_store()\n\n        logger.debug(f\"Search input: {self.search_input}\")\n        logger.debug(f\"Search type: {self.search_type}\")\n        logger.debug(f\"Number of results: {self.number_of_results}\")\n\n        if self.search_input and isinstance(self.search_input, str) and self.search_input.strip():\n            try:\n                search_type = self._map_search_type()\n                search_args = self._build_search_args()\n\n                docs = vector_store.search(query=self.search_input, search_type=search_type, **search_args)\n            except Exception as e:\n                raise ValueError(f\"Error performing search in AstraDBVectorStore: {str(e)}\") from e\n\n            logger.debug(f\"Retrieved documents: {len(docs)}\")\n\n            data = docs_to_data(docs)\n            logger.debug(f\"Converted documents to data: {len(data)}\")\n            self.status = data\n            return data\n        else:\n            logger.debug(\"No search input provided. Skipping search.\")\n            return []\n\n    def get_retriever_kwargs(self):\n        search_args = self._build_search_args()\n        return {\n            \"search_type\": self._map_search_type(),\n            \"search_kwargs\": search_args,\n        }\n",
                    "collection_indexing_policy_AstraVectorStoreComponent-86DpN": "",
                    "collection_name_AstraVectorStoreComponent-86DpN": "langflow",
                    "metadata_indexing_exclude_AstraVectorStoreComponent-86DpN": "",
                    "metadata_indexing_include_AstraVectorStoreComponent-86DpN": "",
                    "metric_AstraVectorStoreComponent-86DpN": "",
                    "namespace_AstraVectorStoreComponent-86DpN": "",
                    "number_of_results_AstraVectorStoreComponent-86DpN": 4,
                    "pre_delete_collection_AstraVectorStoreComponent-86DpN": false,
                    "search_filter_AstraVectorStoreComponent-86DpN": {},
                    "search_input_AstraVectorStoreComponent-86DpN": "",
                    "search_score_threshold_AstraVectorStoreComponent-86DpN": 0,
                    "search_type_AstraVectorStoreComponent-86DpN": "Similarity",
                    "setup_mode_AstraVectorStoreComponent-86DpN": "Sync",
                    "token_AstraVectorStoreComponent-86DpN": "",
                    "chunk_size_OpenAIEmbeddings-MawhJ": 1000,
                    "client_OpenAIEmbeddings-MawhJ": "",
                    "code_OpenAIEmbeddings-MawhJ": "from langchain_openai.embeddings.base import OpenAIEmbeddings\n\nfrom langflow.base.embeddings.model import LCEmbeddingsModel\nfrom langflow.base.models.openai_constants import OPENAI_EMBEDDING_MODEL_NAMES\nfrom langflow.field_typing import Embeddings\nfrom langflow.io import BoolInput, DictInput, DropdownInput, FloatInput, IntInput, MessageTextInput, SecretStrInput\n\n\nclass OpenAIEmbeddingsComponent(LCEmbeddingsModel):\n    display_name = \"OpenAI Embeddings\"\n    description = \"Generate embeddings using OpenAI models.\"\n    icon = \"OpenAI\"\n    name = \"OpenAIEmbeddings\"\n\n    inputs = [\n        DictInput(\n            name=\"default_headers\",\n            display_name=\"Default Headers\",\n            advanced=True,\n            info=\"Default headers to use for the API request.\",\n        ),\n        DictInput(\n            name=\"default_query\",\n            display_name=\"Default Query\",\n            advanced=True,\n            info=\"Default query parameters to use for the API request.\",\n        ),\n        IntInput(name=\"chunk_size\", display_name=\"Chunk Size\", advanced=True, value=1000),\n        MessageTextInput(name=\"client\", display_name=\"Client\", advanced=True),\n        MessageTextInput(name=\"deployment\", display_name=\"Deployment\", advanced=True),\n        IntInput(name=\"embedding_ctx_length\", display_name=\"Embedding Context Length\", advanced=True, value=1536),\n        IntInput(name=\"max_retries\", display_name=\"Max Retries\", value=3, advanced=True),\n        DropdownInput(\n            name=\"model\",\n            display_name=\"Model\",\n            advanced=False,\n            options=OPENAI_EMBEDDING_MODEL_NAMES,\n            value=\"text-embedding-3-small\",\n        ),\n        DictInput(name=\"model_kwargs\", display_name=\"Model Kwargs\", advanced=True),\n        SecretStrInput(name=\"openai_api_base\", display_name=\"OpenAI API Base\", advanced=True),\n        SecretStrInput(name=\"openai_api_key\", display_name=\"OpenAI API Key\", value=\"OPENAI_API_KEY\"),\n        SecretStrInput(name=\"openai_api_type\", display_name=\"OpenAI API Type\", advanced=True),\n        MessageTextInput(name=\"openai_api_version\", display_name=\"OpenAI API Version\", advanced=True),\n        MessageTextInput(\n            name=\"openai_organization\",\n            display_name=\"OpenAI Organization\",\n            advanced=True,\n        ),\n        MessageTextInput(name=\"openai_proxy\", display_name=\"OpenAI Proxy\", advanced=True),\n        FloatInput(name=\"request_timeout\", display_name=\"Request Timeout\", advanced=True),\n        BoolInput(name=\"show_progress_bar\", display_name=\"Show Progress Bar\", advanced=True),\n        BoolInput(name=\"skip_empty\", display_name=\"Skip Empty\", advanced=True),\n        MessageTextInput(\n            name=\"tiktoken_model_name\",\n            display_name=\"TikToken Model Name\",\n            advanced=True,\n        ),\n        BoolInput(\n            name=\"tiktoken_enable\",\n            display_name=\"TikToken Enable\",\n            advanced=True,\n            value=True,\n            info=\"If False, you must have transformers installed.\",\n        ),\n        IntInput(\n            name=\"dimensions\",\n            display_name=\"Dimensions\",\n            info=\"The number of dimensions the resulting output embeddings should have. Only supported by certain models.\",\n            advanced=True,\n        ),\n    ]\n\n    def build_embeddings(self) -> Embeddings:\n        return OpenAIEmbeddings(\n            tiktoken_enabled=self.tiktoken_enable,\n            default_headers=self.default_headers,\n            default_query=self.default_query,\n            allowed_special=\"all\",\n            disallowed_special=\"all\",\n            chunk_size=self.chunk_size,\n            deployment=self.deployment,\n            embedding_ctx_length=self.embedding_ctx_length,\n            max_retries=self.max_retries,\n            model=self.model,\n            model_kwargs=self.model_kwargs,\n            base_url=self.openai_api_base,\n            api_key=self.openai_api_key,\n            openai_api_type=self.openai_api_type,\n            api_version=self.openai_api_version,\n            organization=self.openai_organization,\n            openai_proxy=self.openai_proxy,\n            timeout=self.request_timeout or None,\n            show_progress_bar=self.show_progress_bar,\n            skip_empty=self.skip_empty,\n            tiktoken_model_name=self.tiktoken_model_name,\n            dimensions=self.dimensions or None,\n        )\n",
                    "default_headers_OpenAIEmbeddings-MawhJ": {},
                    "default_query_OpenAIEmbeddings-MawhJ": {},
                    "deployment_OpenAIEmbeddings-MawhJ": "",
                    "dimensions_OpenAIEmbeddings-MawhJ": null,
                    "embedding_ctx_length_OpenAIEmbeddings-MawhJ": 1536,
                    "max_retries_OpenAIEmbeddings-MawhJ": 3,
                    "model_OpenAIEmbeddings-MawhJ": "text-embedding-3-small",
                    "model_kwargs_OpenAIEmbeddings-MawhJ": {},
                    "openai_api_base_OpenAIEmbeddings-MawhJ": "",
                    "openai_api_key_OpenAIEmbeddings-MawhJ": "",
                    "openai_api_type_OpenAIEmbeddings-MawhJ": "",
                    "openai_api_version_OpenAIEmbeddings-MawhJ": "",
                    "openai_organization_OpenAIEmbeddings-MawhJ": "",
                    "openai_proxy_OpenAIEmbeddings-MawhJ": "",
                    "request_timeout_OpenAIEmbeddings-MawhJ": null,
                    "show_progress_bar_OpenAIEmbeddings-MawhJ": false,
                    "skip_empty_OpenAIEmbeddings-MawhJ": false,
                    "tiktoken_enable_OpenAIEmbeddings-MawhJ": true,
                    "tiktoken_model_name_OpenAIEmbeddings-MawhJ": ""
                },
                "Memory-Vna9Y": {
                    "n_messages": 20,
                    "order": "Ascending",
                    "sender": "Machine and User",
                    "sender_name": "",
                    "session_id": "",
                    "template": "{sender_name}: {text}"
                },
                "TextInput-NB6T3": {
                    "input_value": displayName
                },
                "TextInput-l4xtl": {
                    "input_value": userId
                },
                "TextInput-obPky": {
                    "input_value": input
                }
            }
        }),
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data: any = await response.json();
    if (data.outputs && data.outputs.length > 0) {
        if (data.outputs[0].outputs && data.outputs[0].outputs.length > 0) {
            if (data.outputs[0].outputs[0].results?.message?.text) {
                return data.outputs[0].outputs[0].results?.message?.text;
            }
        }
    }
    return 'No response from LLM';
}

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
